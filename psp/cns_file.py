import struct
from pathlib import Path

__all__ = [
    'CNSFile',
    'decompress',
]

def decompress(infile, outfile=None):
    """Decompresses an input CNS file into an output file.

    If outfile is not given, it will try to append the extension found in the file to the infile, so make sure infile is a path.
    """
    cns = CNSFile(infile)
    if outfile is None:
        outfile = Path(infile).with_suffix('.' + cns.ext)
    cns.save(outfile)



class CNSFile:
    MAGIC = b'@CNS'
    FORMAT = '<4sI4sIIII'
    assert struct.calcsize(FORMAT) == 0x1C
    
    def __init__(self, file):
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as file:
                self.init_file(file)
                assert not file.read()
        else:
            self.init_file(file)
    
    def init_file(self, file):
        start = file.tell()
        head_raw = file.read(0x20)
        assert head_raw[:4] == self.MAGIC
        head = struct.unpack(self.FORMAT, head_raw[4:])
        [
            ext,    # Maybe usable as the file extension.
            size,   # Decompressed size
            UNK_0C,
            UNK_10,
            data_ptr,
            UNK_18,
            UNK_1C,
        ] = head
        if ext == b'ptx\0':
            ...
        elif ext == b'ptx\0':
            ...
        else:
            raise ValueError("Unknown inner type: %r" % innertype)
        assert UNK_0C == b'101\0'
        assert UNK_10 == 0
        assert data_ptr == 0x20
        assert UNK_18 == 0
        assert UNK_1C == 0

        assert start + data_ptr == file.tell()
        buf = bytearray(size)
        char = file.read(1)[0]
        i = 0
        while char:
            if char < 0x80:  #Copy raw bytes from stream.
                n = char
                buf[i:i+n] = file.read(n)
                i += n
            else:  #Copy from buffer.
                x = char - 0x80 + 3
                y = file.read(1)[0] + 1
                if x <= y:  #There are enough bytes to copy.
                    buf[i:i+x] = buf[i-y:i-y+x]
                else:  # Source and destination overlap.
                    back = buf[i-y:i]
                    q, r = divmod(x, len(back))
                    buf[i:i+x] = back*q + back[:r]
                i += x
            char = file.read(1)[0]
        assert i == len(buf)
        self.ext = ext.decode().rstrip('\0')
        self.data = buf

    def save(self, file):
        if isinstance(file, (str, Path)):
            with open(file, 'wb') as file:
                file.write(self.data)
        else:
            file.write(self.data)



