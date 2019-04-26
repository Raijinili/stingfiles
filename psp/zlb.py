#zlb.py
import zlib

def decompress(fpath, outpath):
    with open(fpath, 'rb') as file:
        data = file.read()
    assert data[:4] == b'zlb\0'
    data = zlib.decompress(data[8:])
    if outpath is None:
        outpath = fpath + '.out'
    with open(outpath, 'wb') as file:
        file.write(data)

