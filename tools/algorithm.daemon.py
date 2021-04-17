    
def _gen_crc(crc):
    for j in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xEDB88320
        else:
            crc >>= 1
    return crc

_crctable = list(map(_gen_crc, range(256)))

# print(_crctable)

def _ZipDecrypter(pwd):
    key0 = 305419896
    key1 = 591751049
    key2 = 878082192

    crctable = _crctable

    def crc32(ch, crc):
        """Compute the CRC32 primitive on one byte."""
        return (crc >> 8) ^ crctable[(crc ^ ch) & 0xFF]

    def update_keys(c):
        nonlocal key0, key1, key2
        key0 = crc32(c, key0)
        key1 = (key1 + (key0 & 0xFF)) & 0xFFFFFFFF
        key1 = (key1 * 134775813 + 1) & 0xFFFFFFFF
        key2 = crc32(key1 >> 24, key2)
    
    for p in pwd:
        update_keys(p)
    
    # print(key0, key1, key2)
    
    def decrypter(data):
        """Decrypt a bytes object."""
        last = None
        for c in data:
            k = key2 | 2
            c ^= ((k * (k^1)) >> 8) & 0xFF
            update_keys(c)
            last = c
        return last
  

    return decrypter


if __name__ == '__main__':
    decrypt = _ZipDecrypter(bytes("123456", "ascii"))
    print(decrypt(bytes([0x2d, 0x40, 0xa2, 0x29, 0x41, 0x65, 0xf5, 0x78, 0xce, 0x92, 0x23, 0x40])))

    
    

    