import numpy as np
from scipy import signal

from src.symbols import Symbols

import numpy as np

# n = 6
#
# # Coefficients of the polynomial x^2 + 2x + 1
# p = Symbols.CIRCLE.quarter_probabilities
#
# # Determine the required length for the FFT and pad the array
# # Length must be at least 2*len(p) - 1, which is 5.
# # Let's use a power of 2 for efficiency, e.g., 8.
# fft_length = 100
# padded_p = np.pad(p, (0, fft_length - len(p)))
#
# # Compute the FFT, take the element-wise square root, and then the inverse FFT
# fft_p = np.fft.fft(padded_p)
# sqrt_fft_p = np.pow(fft_p, 1/n)
# q_padded = np.fft.ifft(sqrt_fft_p)
#
# # Extract the real part and crop to the correct length (len(p))
# q = q_padded.real[:len(p)]
#
# print("Original polynomial coefficients (p):", p)
# print("Square root polynomial coefficients (q):", q)
#
# # Verify the result by convolving q with itself
# verification = q
#
# for i in range(1,n):
#     verification = np.convolve(verification, q)[:len(p)]
#
# print(verification)

p = np.array([0,0,0,1])
q = Symbols.CIRCLE.