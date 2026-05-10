import zlib

import numpy as np
import qrcode


QR_BYTE_CAPACITY_L = 2953


def adaptive_downsample(sequence, target_length):
    """Select evenly spaced points from a 1D key sequence."""
    sequence = np.asarray(sequence, dtype=np.float32).reshape(-1)
    if sequence.size <= target_length:
        return sequence

    indices = np.linspace(0, sequence.size - 1, target_length).astype(int)
    return sequence[indices]


def quantize_sequence(sequence):
    """Map a 1D analog key sequence to 8-bit symbols for QR storage."""
    sequence = np.asarray(sequence, dtype=np.float32).reshape(-1)
    if sequence.size == 0:
        raise ValueError("target_sequence must contain at least one sample")

    sequence = np.nan_to_num(sequence, nan=0.0)
    min_val, max_val = float(sequence.min()), float(sequence.max())
    if max_val == min_val:
        return np.zeros(sequence.shape, dtype=np.uint8)

    normalized = (sequence - min_val) / (max_val - min_val)
    return np.round(normalized * 255).astype(np.uint8)


def compress_sequence(sequence):
    """Compress quantized key bytes before QR encoding."""
    quantized = quantize_sequence(sequence)
    return zlib.compress(quantized.tobytes())


def generate_qr_from_compressed_data(compressed_data, file_name="compressed_qr.png"):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(compressed_data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(file_name)
    print(f"QR code generated and saved: {file_name}")


def generate_qr_for_key_sequence(
    target_sequence,
    output_prefix="compressed_qr",
    max_target_length=2000,
    min_target_length=500,
    step_length=100,
):
    """
    Encode one 1D chaotic key sequence into a QR code.

    The sequence is downsampled only if the compressed 8-bit key payload exceeds
    the single-QR byte capacity used in this demo.
    """
    sequence = np.asarray(target_sequence, dtype=np.float32).reshape(-1)
    print("Original sequence length:", sequence.size)

    for target_length in range(max_target_length, min_target_length - 1, -step_length):
        sampled_sequence = adaptive_downsample(sequence, target_length)
        compressed_data = compress_sequence(sampled_sequence)
        compressed_size = len(compressed_data)

        print(f"Candidate points: {sampled_sequence.size}, compressed size: {compressed_size} bytes")
        if compressed_size <= QR_BYTE_CAPACITY_L:
            file_name = f"{output_prefix}_{sampled_sequence.size}.png"
            generate_qr_from_compressed_data(compressed_data, file_name=file_name)
            return file_name

    raise ValueError("Compressed key payload exceeds the single-QR capacity in this demo.")


def main():
    # Replace this placeholder with one measured 1D chaotic key sequence.
    target_sequence = np.asarray([], dtype=np.float32)
    generate_qr_for_key_sequence(target_sequence)


if __name__ == "__main__":
    main()
