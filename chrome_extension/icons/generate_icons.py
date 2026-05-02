#!/usr/bin/env python3
"""Generate simple PNG icons for Chrome extension"""
import struct
import zlib

def create_png(width, height, color_rgb):
    """Create a simple solid-color PNG with a circle"""
    
    def png_chunk(chunk_type, data):
        chunk_len = struct.pack('>I', len(data))
        chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
        return chunk_len + chunk_type + data + chunk_crc
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b'IHDR', ihdr_data)
    
    # Create image data with a circle
    raw_data = b''
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 2 - 1
    
    for y in range(height):
        raw_data += b'\x00'  # Filter byte
        for x in range(width):
            # Check if pixel is inside circle
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            if dist <= radius:
                # Inner circle (lighter)
                if dist <= radius * 0.5:
                    r, g, b = 255, 215, 0  # Gold center
                else:
                    # Middle ring
                    if dist <= radius * 0.75:
                        r, g, b = 80, 200, 120  # Green
                    else:
                        # Outer ring
                        r, g, b = color_rgb
            else:
                # Transparent background
                r, g, b = 255, 255, 255
            
            raw_data += bytes([r, g, b])
    
    # Compress image data
    compressed = zlib.compress(raw_data, 9)
    idat = png_chunk(b'IDAT', compressed)
    
    # IEND chunk
    iend = png_chunk(b'IEND', b'')
    
    return signature + ihdr + idat + iend

# Generate icons
colors = [(74, 144, 217), (74, 144, 217), (74, 144, 217)]  # Blue base
sizes = [16, 48, 128]

for size, color in zip(sizes, colors):
    png_data = create_png(size, size, color)
    with open(f'icon{size}.png', 'wb') as f:
        f.write(png_data)
    print(f'Generated icon{size}.png')

print('Done!')
