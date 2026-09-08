import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class Symbol:
    """Represents a detected symbol with its bounding box."""
    name: str
    bbox: List[float]  # [ymin, ymax, xmin, xmax]
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        """Returns (x_center, y_center)."""
        x_center = (self.bbox[2] + self.bbox[3]) / 2
        y_center = (self.bbox[0] + self.bbox[1]) / 2
        return (x_center, y_center)

    @property
    def width(self) -> float:
        return self.bbox[3] - self.bbox[2]

    @property
    def height(self) -> float:
        return self.bbox[1] - self.bbox[0]


@dataclass
class Quarter:
    """Represents a quarter card in the grid."""
    bbox: List[float]  # [ymin, ymax, xmin, xmax]
    x: int = -1
    y: int = -1
    symbols: List[Symbol] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns (x_center, y_center)."""
        x_center = (self.bbox[2] + self.bbox[3]) / 2
        y_center = (self.bbox[0] + self.bbox[1]) / 2
        return (x_center, y_center)

    @property
    def width(self) -> float:
        return self.bbox[3] - self.bbox[2]

    @property
    def height(self) -> float:
        return self.bbox[1] - self.bbox[0]

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this quarter's bounding box."""
        xmin, xmax = self.bbox[2], self.bbox[3]
        ymin, ymax = self.bbox[0], self.bbox[1]
        return xmin <= x <= xmax and ymin <= y <= ymax

class CardGameScorer:
    """Calculates points for the card game based on object detection predictions."""

    def __init__(
            self,
            prediction: Dict,
            iou_threshold_different: float = 0.75,
            iou_threshold_same: float = 0.50,
            min_confidence: float = 0.5,
            rotate_180: bool = True,
            size_tolerance: float = 0.3,
            position_threshold_factor: float = 0.4,
            spacing_tolerance: float = 0.5
    ):
        """
        Initialize with prediction dictionary.

        Args:
            prediction: Dictionary with 'bboxes', 'displayNames', 'confidences'
            iou_threshold_different: IoU threshold for filtering different symbols (default: 0.75)
            iou_threshold_same: IoU threshold for filtering same symbols (default: 0.50)
            min_confidence: Minimum confidence threshold for predictions (default: 0.5)
            rotate_180: Whether to rotate bounding boxes 180 degrees (default: True)
            size_tolerance: Tolerance for quarter size variation (±30% = 0.3) (default: 0.3)
            position_threshold_factor: Factor for position threshold calculation (default: 0.4)
            spacing_tolerance: Tolerance for spacing consistency within rows (default: 0.5)
        """
        self.prediction = prediction
        self.iou_threshold_different = iou_threshold_different
        self.iou_threshold_same = iou_threshold_same
        self.min_confidence = min_confidence
        self.rotate_180 = rotate_180
        self.size_tolerance = size_tolerance
        self.position_threshold_factor = position_threshold_factor
        self.spacing_tolerance = spacing_tolerance
        self.quarters: List[Quarter] = []
        self.symbols: List[Symbol] = []
        self.grid: Dict[Tuple[int, int], Quarter] = {}

        self._parse_predictions()
        self._filter_overlapping_symbols()

    def _rotate_bbox_180(self, bbox: List[float]) -> List[float]:
        """
        Mirror a bounding box around the y-axis (horizontal flip).

        Args:
            bbox: Bounding box in format [ymin, ymax, xmin, xmax]

        Returns:
            Mirrored bounding box in format [ymin, ymax, xmin, xmax]
        """
        ymin, ymax, xmin, xmax = bbox

        # Mirror around y-axis: new_x = 1 - old_x, keep y unchanged
        mirrored_ymin = 1.0 - ymin
        mirrored_ymax = 1.0 - ymax

        return [mirrored_ymin/2, mirrored_ymax/2, xmin, xmax]

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1, bbox2: Bounding boxes in format [ymin, ymax, xmin, xmax]

        Returns:
            IoU value between 0 and 1
        """
        # Extract coordinates
        y1_min, y1_max, x1_min, x1_max = bbox1
        y2_min, y2_max, x2_min, x2_max = bbox2

        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _filter_overlapping_symbols(self):
        """
        Filter overlapping symbols based on total overlap with all other symbols.
        Keeps symbols with lowest total overlap when confidence is similar.
        """
        if not self.symbols:
            return

        # Calculate total overlap for each symbol with all others
        symbol_data = []
        for i, symbol in enumerate(self.symbols):
            total_overlap = 0

            for j, other_symbol in enumerate(self.symbols):
                if i != j:  # Don't compare with itself
                    iou = self._calculate_iou(symbol.bbox, other_symbol.bbox)

                    # Weight the overlap based on symbol type similarity
                    if symbol.name == other_symbol.name:
                        threshold = self.iou_threshold_same
                    else:
                        threshold = self.iou_threshold_different

                    # Only count significant overlaps
                    if iou > threshold:
                        total_overlap += iou

            symbol_data.append((total_overlap, symbol.confidence, i, symbol))

        # Sort by total overlap (ascending), then by confidence (descending)
        symbol_data.sort(key=lambda x: (x[0], -x[1]))

        # Keep symbols with lowest total overlap
        filtered_symbols = []
        processed_indices = set()

        for total_overlap, confidence, index, symbol in symbol_data:
            if index not in processed_indices:
                # Check if this symbol conflicts with already kept symbols
                should_keep = True

                for kept_symbol in filtered_symbols:
                    iou = self._calculate_iou(symbol.bbox, kept_symbol.bbox)

                    if symbol.name == kept_symbol.name:
                        threshold = self.iou_threshold_same
                    else:
                        threshold = self.iou_threshold_different

                    if iou > threshold:
                        should_keep = False
                        break

                if should_keep:
                    filtered_symbols.append(symbol)
                    processed_indices.add(index)

        removed_count = len(self.symbols) - len(filtered_symbols)
        if removed_count > 0:
            print(f"Filtered {removed_count} overlapping symbols based on total overlap analysis")

        self.symbols = filtered_symbols

    def _parse_predictions(self):
        """Parse predictions into quarters and symbols, filtering by confidence."""
        bboxes = self.prediction['bboxes']
        names = self.prediction['displayNames']
        confidences = self.prediction['confidences']

        filtered_count = 0
        for bbox, name, conf in zip(bboxes, names, confidences):
            if conf < self.min_confidence:
                filtered_count += 1
                continue

            # Apply 180-degree rotation if enabled
            if self.rotate_180:
                final_bbox = self._rotate_bbox_180(bbox)
            else:
                final_bbox = bbox

            if name == 'quarter':
                self.quarters.append(Quarter(bbox=final_bbox))
            else:
                self.symbols.append(Symbol(name=name, bbox=final_bbox, confidence=conf))

        if filtered_count > 0:
            print(f"Filtered {filtered_count} predictions below confidence threshold {self.min_confidence}")

    def visualize_bboxes(self, figsize=(12, 8)):
        """Visualize bounding boxes - quarters in black, symbols in green."""
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_aspect('equal')

        # Draw quarters with black outline
        for quarter in self.quarters:
            ymin, ymax, xmin, xmax = quarter.bbox
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

        # Draw symbols with green boxes
        for symbol in self.symbols:
            ymin, ymax, xmin, xmax = symbol.bbox
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height,
                                     linewidth=1, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            x_center, y_center = symbol.center
            ax.text(x_center, y_center, symbol.name, fontsize=6, ha='center', color='green')

        plt.title("Detected Objects: Quarters (black) and Symbols (green)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def create_grid(self):
        """Create grid from quarters with x,y coordinates, rejecting outliers."""
        if not self.quarters:
            return

        # Calculate median size to identify outliers
        widths = [q.width for q in self.quarters]
        heights = [q.height for q in self.quarters]
        median_width = np.median(widths)
        median_height = np.median(heights)

        # Filter out quarters with significantly different sizes (configurable tolerance)
        valid_quarters = []

        for quarter in self.quarters:
            width_ratio = abs(quarter.width - median_width) / median_width
            height_ratio = abs(quarter.height - median_height) / median_height

            if width_ratio <= self.size_tolerance and height_ratio <= self.size_tolerance:
                valid_quarters.append(quarter)

        removed_size = len(self.quarters) - len(valid_quarters)
        if removed_size > 0:
            print(f"Removed {removed_size} quarters due to size mismatch (tolerance: {self.size_tolerance})")

        self.quarters = valid_quarters

        if not self.quarters:
            print("No valid quarters remaining after size filtering")
            return

        # Sort quarters by position (y first, then x)
        sorted_quarters = sorted(self.quarters, key=lambda q: (q.center[1], q.center[0]))

        # Group quarters into rows based on y-coordinate proximity
        rows = []
        current_row = [sorted_quarters[0]]
        y_threshold = median_height * self.position_threshold_factor  # Configurable threshold

        for quarter in sorted_quarters[1:]:
            if abs(quarter.center[1] - current_row[0].center[1]) < y_threshold:
                current_row.append(quarter)
            else:
                rows.append(sorted(current_row, key=lambda q: q.center[0]))
                current_row = [quarter]
        rows.append(sorted(current_row, key=lambda q: q.center[0]))

        # Validate grid structure - remove quarters that don't align properly
        validated_quarters = []
        x_threshold = median_width * self.position_threshold_factor  # Configurable threshold

        for row in rows:
            if len(row) == 1:
                # Single quarters might be outliers, check if they align with other rows
                quarter = row[0]
                aligned = False

                for other_row in rows:
                    if other_row != row and len(other_row) > 1:
                        for other_quarter in other_row:
                            if abs(quarter.center[0] - other_quarter.center[0]) < x_threshold:
                                aligned = True
                                break
                        if aligned:
                            break

                if aligned or len(rows) == 1:  # Keep if aligned or if it's the only row
                    validated_quarters.extend(row)
            else:
                # Check for consistent spacing within the row
                if len(row) > 1:
                    spacings = []
                    for i in range(1, len(row)):
                        spacings.append(row[i].center[0] - row[i-1].center[0])

                    median_spacing = np.median(spacings)
                    valid_row = []

                    # Keep quarters that maintain consistent spacing (configurable tolerance)
                    valid_row.append(row[0])  # Always keep first
                    for i in range(1, len(row)):
                        spacing = row[i].center[0] - row[i-1].center[0]
                        if abs(spacing - median_spacing) / median_spacing < self.spacing_tolerance:
                            valid_row.append(row[i])

                    validated_quarters.extend(valid_row)

        removed_position = len(self.quarters) - len(validated_quarters)
        if removed_position > 0:
            print(f"Removed {removed_position} quarters due to poor grid alignment (position factor: {self.position_threshold_factor}, spacing tolerance: {self.spacing_tolerance})")

        self.quarters = validated_quarters

        if not self.quarters:
            print("No valid quarters remaining after grid validation")
            return

        # Rebuild rows with validated quarters
        sorted_quarters = sorted(self.quarters, key=lambda q: (q.center[1], q.center[0]))
        rows = []
        current_row = [sorted_quarters[0]]

        for quarter in sorted_quarters[1:]:
            if abs(quarter.center[1] - current_row[0].center[1]) < y_threshold:
                current_row.append(quarter)
            else:
                rows.append(sorted(current_row, key=lambda q: q.center[0]))
                current_row = [quarter]
        rows.append(sorted(current_row, key=lambda q: q.center[0]))

        # Assign grid coordinates
        for y, row in enumerate(rows):
            for x, quarter in enumerate(row):
                quarter.x = x
                quarter.y = y
                self.grid[(x, y)] = quarter

        print(f"Created grid with {len(rows)} rows and max {max(len(row) for row in rows)} columns")
        print(f"Final quarters: {len(self.quarters)}")

    def assign_symbols_to_quarters(self):
        """Assign each symbol to the quarter it's located in."""
        for symbol in self.symbols:
            x_center, y_center = symbol.center
            for quarter in self.quarters:
                if quarter.contains_point(x_center, y_center):
                    quarter.symbols.append(symbol)
                    break

    def process_arrows(self):
        """Process arrows by copying symbols from pointed quarters."""
        arrow_directions = {
            'arrow_left': (-1, 0),
            'arrow_right': (1, 0),
            'arrow_up': (0, -1),
            'arrow_down': (0, 1)
        }

        for quarter in self.quarters:
            arrows_to_remove = []
            symbols_to_add = []

            for symbol in quarter.symbols:
                if symbol.name in arrow_directions:
                    dx, dy = arrow_directions[symbol.name]
                    target_coords = (quarter.x + dx, quarter.y + dy)

                    if target_coords in self.grid:
                        target_quarter = self.grid[target_coords]
                        # Copy non-arrow symbols
                        for target_symbol in target_quarter.symbols:
                            if target_symbol.name not in arrow_directions:
                                symbols_to_add.append(target_symbol)

                    arrows_to_remove.append(symbol)

            # Remove arrows and add copied symbols
            for arrow in arrows_to_remove:
                quarter.symbols.remove(arrow)
            quarter.symbols.extend(symbols_to_add)

    def count_symbols(self) -> Dict[str, int]:
        """Count total number of each symbol type across all quarters."""
        symbol_counts = {}

        for quarter in self.quarters:
            for symbol in quarter.symbols:
                symbol_counts[symbol.name] = symbol_counts.get(symbol.name, 0) + 1

        return symbol_counts

    def calculate_score(self) -> Dict:
        """
        Main method to process everything and calculate final score.

        Returns:
            Dictionary with symbol counts and total
        """
        self.create_grid()
        self.assign_symbols_to_quarters()
        self.process_arrows()

        symbol_counts = self.count_symbols()
        total_symbols = sum(symbol_counts.values())

        return {
            'symbol_counts': symbol_counts,
            'total_symbols': total_symbols,
            'quarters_count': len(self.quarters)
        }

    def print_grid_state(self):
        """Print the current state of the grid for debugging."""
        print("\nGrid State:")
        for (x, y), quarter in sorted(self.grid.items()):
            symbols_str = ", ".join([s.name for s in quarter.symbols])
            print(f"Quarter ({x}, {y}): {symbols_str if symbols_str else 'empty'}")
if __name__ == "__main__":
    prediction = {'bboxes': [[0.719872773, 0.883248091, 0.210903585, 0.309707463], [0.0706562549, 0.231251344, 0.213013276, 0.308854938], [0.568326235, 0.726024389, 0.119086429, 0.215034232], [0.0749601796, 0.233899325, 0.304783672, 0.397995621], [0.236580744, 0.392235637, 0.302816182, 0.39877063], [0.721787155, 0.883975863, 0.120224133, 0.216629103], [0.388447642, 0.548267484, 0.21390605, 0.311274], [0.238918886, 0.393207759, 0.394162476, 0.48670572], [0.548319221, 0.704029799, 0.305405766, 0.397427082], [0.564128339, 0.721548319, 0.211035043, 0.306527168], [0.551647425, 0.705738723, 0.479432344, 0.570964813], [0.390073299, 0.546453893, 0.306301206, 0.398346424], [0.397572666, 0.550402462, 0.478329957, 0.570125759], [0.551221609, 0.706699312, 0.392088205, 0.484716892], [0.230052471, 0.38587153, 0.214996159, 0.311704099], [0.401214838, 0.553034544, 0.392264783, 0.485057861], [0.304522961, 0.382288814, 0.217388302, 0.263724595], [0.149865672, 0.225854844, 0.215959966, 0.260850936], [0.0774596557, 0.15290314, 0.351757824, 0.395309448], [0.561769783, 0.640690684, 0.258172929, 0.303524166], [0.232696936, 0.31061992, 0.263112247, 0.309738576], [0.280215144, 0.356251389, 0.346051246, 0.391828746], [0.147022933, 0.225794584, 0.308526665, 0.353312641], [0.473568857, 0.544811547, 0.354618609, 0.394952714], [0.470698684, 0.54538542, 0.23734653, 0.281586647], [0.229431391, 0.308872104, 0.279136479, 0.323425919], [0.237457618, 0.316940159, 0.30325, 0.350171924], [0.647763669, 0.71923691, 0.263254821, 0.304739207], [0.389515102, 0.465112805, 0.216999784, 0.262644202], [0.397816122, 0.473161489, 0.394160718, 0.439029604], [0.626390517, 0.702220619, 0.435808033, 0.480559111], [0.315435797, 0.390110344, 0.309392422, 0.352668017], [0.580003679, 0.672482133, 0.498171926, 0.551659], [0.607243419, 0.685252607, 0.143311605, 0.18929325], [0.0745485872, 0.150591925, 0.261734635, 0.30609104], [0.633709729, 0.701108515, 0.394285589, 0.433961391], [0.552752614, 0.627018452, 0.3934578, 0.436949611], [0.554119051, 0.630952895, 0.434923291, 0.480335325], [0.313745558, 0.390956551, 0.394927114, 0.439777225], [0.390361041, 0.555532157, 0.446978897, 0.539193928], [0.600808859, 0.689680636, 0.215374425, 0.267847478], [0.584239662, 0.691788435, 0.31643942, 0.380358785], [0.475582451, 0.55096817, 0.436683804, 0.481097937], [0.550056279, 0.625811338, 0.306342602, 0.350363612], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.80476886, 0.881691456, 0.169101968, 0.213490069], [0.573438, 0.67822051, 0.491559714, 0.553369284], [0.55034256, 0.710722089, 0.446696, 0.540937185], [0.391384244, 0.469095588, 0.262626976, 0.307517678], [0.394458354, 0.473896831, 0.308650434, 0.354866087], [0.728180647, 0.806475818, 0.122798234, 0.16847989], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.594011128, 0.670569301, 0.394179523, 0.437896937], [0.426928371, 0.517572463, 0.49717477, 0.550566673], [0.594011128, 0.670569301, 0.394179523, 0.437896937], [0.593873739, 0.672491252, 0.435084611, 0.482653081], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.552071154, 0.630131543, 0.348721087, 0.393735111], [0.593873739, 0.672491252, 0.435084611, 0.482653081], [0.391384244, 0.469095588, 0.262626976, 0.307517678], [0.6112293, 0.695028722, 0.326058686, 0.375377566], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.391384244, 0.469095588, 0.262626976, 0.307517678], [0.560558915, 0.692134917, 0.487615228, 0.564429879], [0.387851119, 0.465268254, 0.21706897, 0.264065444], [0.286538869, 0.355818629, 0.441276819, 0.482517868], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.393835217, 0.501125276, 0.313987583, 0.378478259], [0.229431391, 0.308872104, 0.279136479, 0.323425919], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.110109329, 0.222907647, 0.213557407, 0.280146301], [0.565522909, 0.674601257, 0.240931869, 0.304574], [0.387851119, 0.465268254, 0.21706897, 0.264065444], [0.450722367, 0.542152, 0.341619223, 0.394004941], [0.726776361, 0.803393602, 0.123902731, 0.167787686], [0.387851119, 0.465268254, 0.21706897, 0.264065444], [0.389983714, 0.465628564, 0.260455281, 0.305312], [0.229431391, 0.308872104, 0.279136479, 0.323425919], [0.390801191, 0.46995455, 0.308757424, 0.353485972], [0.229431391, 0.308872104, 0.279136479, 0.323425919], [0.387851119, 0.465268254, 0.21706897, 0.264065444], [0.555539966, 0.634191, 0.284901619, 0.32840088], [0.600875914, 0.712648511, 0.238031954, 0.304326], [0.239759, 0.314349264, 0.395811051, 0.439381659], [0.504330218, 0.574395478, 0.350259602, 0.393486768], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.726776361, 0.803393602, 0.123902731, 0.167787686], [0.389709949, 0.468662947, 0.281020701, 0.32403335], [0.239759, 0.314349264, 0.395811051, 0.439381659], [0.552752614, 0.627018452, 0.3934578, 0.436949611], [0.0669308454, 0.141735867, 0.338484645, 0.385877728], [0.607243419, 0.685252607, 0.143311605, 0.18929325], [0.473568857, 0.544811547, 0.354618609, 0.394952714], [0.565522909, 0.674601257, 0.240931869, 0.304574], [0.304522961, 0.382288814, 0.217388302, 0.263724595], [0.473568857, 0.544811547, 0.354618609, 0.394952714], [0.550527334, 0.659924686, 0.396805495, 0.460022599], [0.588361621, 0.701103508, 0.412996113, 0.479092777], [0.593873739, 0.672491252, 0.435084611, 0.482653081], [0.389983714, 0.465628564, 0.260455281, 0.305312]], 'confidences': [0.999999762, 0.999999523, 0.999999523, 0.999999166, 0.999999166, 0.999999046, 0.999998808, 0.999998808, 0.999998808, 0.999997497, 0.999996424, 0.999994874, 0.999994636, 0.999992132, 0.999992132, 0.999981284, 0.822463453, 0.793167889, 0.765848, 0.741734803, 0.73829335, 0.71190995, 0.683104, 0.605770946, 0.590439677, 0.578516304, 0.573874652, 0.5397681, 0.523263931, 0.497043788, 0.486626565, 0.477315515, 0.460080534, 0.458181918, 0.455407441, 0.450303435, 0.42861405, 0.425909728, 0.424838036, 0.418363273, 0.40972659, 0.408966511, 0.408392638, 0.384374648, 0.315398484, 0.297224253, 0.289260715, 0.288304657, 0.263907939, 0.246341273, 0.233967677, 0.225927308, 0.219368696, 0.215912536, 0.215044767, 0.210574508, 0.207908273, 0.198136672, 0.184629694, 0.180244863, 0.174787611, 0.172098041, 0.169605941, 0.166822553, 0.16403164, 0.163252056, 0.162874162, 0.153473705, 0.151839316, 0.149325639, 0.142875642, 0.139610171, 0.136337444, 0.134710923, 0.131997138, 0.131356686, 0.129742295, 0.127246946, 0.12693657, 0.124894187, 0.120058686, 0.118836693, 0.118695445, 0.11797896, 0.11568477, 0.114654779, 0.114115082, 0.111313775, 0.110088512, 0.108966589, 0.108405761, 0.107297204, 0.107183456, 0.105026849, 0.10145849, 0.101317331, 0.101197653, 0.100354321, 0.0982771143, 0.0976344123], 'ids': ['8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '8423323945775661056', '4099868303499984896', '929334165831155712', '7270402441168814080', '7270402441168814080', '7270402441168814080', '7558632817320525824', '4964559431955120128', '929334165831155712', '929334165831155712', '7270402441168814080', '7270402441168814080', '929334165831155712', '3811637927348273152', '7270402441168814080', '4099868303499984896', '4964559431955120128', '6117480936561967104', '3811637927348273152', '6117480936561967104', '4964559431955120128', '7558632817320525824', '7270402441168814080', '4099868303499984896', '8423323945775661056', '4099868303499984896', '8423323945775661056', '4964559431955120128', '4964559431955120128', '4964559431955120128', '1794025294286290944', '8423323945775661056', '8423323945775661056', '2658716422741426176', '8711554321927372800', '8711554321927372800', '1794025294286290944', '4964559431955120128', '1794025294286290944', '7558632817320525824', '7270402441168814080', '7558632817320525824', '3811637927348273152', '4099868303499984896', '1505794918134579200', '8711554321927372800', '8711554321927372800', '352873413527732224', '6117480936561967104', '4099868303499984896', '8711554321927372800', '2658716422741426176', '8423323945775661056', '4964559431955120128', '1505794918134579200', '929334165831155712', '7270402441168814080', '6117480936561967104', '8423323945775661056', '4964559431955120128', '1794025294286290944', '4964559431955120128', '1505794918134579200', '4964559431955120128', '1794025294286290944', '7558632817320525824', '4964559431955120128', '929334165831155712', '4964559431955120128', '929334165831155712', '7270402441168814080', '929334165831155712', '352873413527732224', '8711554321927372800', '4964559431955120128', '7270402441168814080', '4099868303499984896', '6117480936561967104', '8423323945775661056', '352873413527732224', '4964559431955120128', '8423323945775661056', '4099868303499984896', '929334165831155712', '7270402441168814080'], 'displayNames': ['quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'quarter', 'rum', 'coin', 'spyglass', 'spyglass', 'spyglass', 'shark', 'anchor', 'coin', 'coin', 'spyglass', 'spyglass', 'coin', 'parrot', 'spyglass', 'rum', 'anchor', 'map', 'parrot', 'map', 'anchor', 'shark', 'spyglass', 'rum', 'quarter', 'rum', 'quarter', 'anchor', 'anchor', 'anchor', 'rat', 'quarter', 'quarter', 'arrow_left', 'kraken', 'kraken', 'rat', 'anchor', 'rat', 'shark', 'spyglass', 'shark', 'parrot', 'rum', 'arrow_down', 'kraken', 'kraken', 'arrow_right', 'map', 'rum', 'kraken', 'arrow_left', 'quarter', 'anchor', 'arrow_down', 'coin', 'spyglass', 'map', 'quarter', 'anchor', 'rat', 'anchor', 'arrow_down', 'anchor', 'rat', 'shark', 'anchor', 'coin', 'anchor', 'coin', 'spyglass', 'coin', 'arrow_right', 'kraken', 'anchor', 'spyglass', 'rum', 'map', 'quarter', 'arrow_right', 'anchor', 'quarter', 'rum', 'coin', 'spyglass']}
    scorer = CardGameScorer(
        prediction,
        iou_threshold_different=0.80,  # More strict for different symbols
        iou_threshold_same=0.40,  # More lenient for same symbols
        min_confidence=0.4,  # Filter out predictions below 40% confidence
        rotate_180=True,  # Rotate bounding boxes 180 degrees
        size_tolerance=0.25,  # Stricter size validation (±25%)
        position_threshold_factor=0.35,  # More strict position alignment
        spacing_tolerance=0.4  # More strict spacing consistency
    )

    # Step 1: Visualize bboxes
    scorer.visualize_bboxes()

    # Step 2-5: Calculate final score
    result = scorer.calculate_score()

    # Display results
    print(f"\nFinal Results:")
    print(f"Total symbols: {result['total_symbols']}")
    print(f"Quarters: {result['quarters_count']}")
    print(f"\nSymbol breakdown:")
    for symbol, count in sorted(result['symbol_counts'].items()):
        print(f"  {symbol}: {count}")

    # Optional: Print detailed grid state
    scorer.print_grid_state()
