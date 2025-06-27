import qrcode
import pickle
import zlib
import numpy as np
import math

def compress_sequence(sequence):
    bin_data = pickle.dumps(sequence)
    compressed_data = zlib.compress(bin_data)
    return compressed_data

def adaptive_downsample(sequence, target_length):
    original_len = len(sequence)
    if original_len <= target_length:
        return sequence

    step = original_len / target_length
    sampled_sequence = [sequence[int(i * step)] for i in range(target_length)]
    return sampled_sequence

def generate_qr_from_compressed_data(compressed_data, file_name='compressed_qr.png'):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4
    )

    qr.add_data(compressed_data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(file_name)
    print(f"QR code generated and saved: {file_name}")

def main():
    np.random.seed(0)
    sequence = [target_sequence]  # Replace with your target sequence
    print("Original sequence length:", len(sequence))

    max_target_length = 2000
    min_target_length = 500
    step_length = 100

    for target_length in range(max_target_length, min_target_length - 1, -step_length):
        sampled_sequence = adaptive_downsample(sequence, target_length)
        print(f"Downsampled length: {len(sampled_sequence)}")

        compressed_data = compress_sequence(sampled_sequence)
        compressed_size = len(compressed_data)
        print(f"Compressed data size: {compressed_size} bytes")

        if compressed_size <= 2953:  # QR code capacity limit
            print(f"Success: Data fits in single QR code, target points: {target_length}")
            generate_qr_from_compressed_data(compressed_data, file_name=f'compressed_qr_{target_length}.png')
            break
    else:
        print("Failed: Data exceeds QR code capacity even at minimum target points.")

if __name__ == "__main__":
    main()