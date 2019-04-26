# fct_file.py
import struct
import os
import os.path

def unpackFCT(fpath):
    with open(fpath, 'rb') as file:
        head_raw = file.read(0x10)
        assert head_raw[:4] == b'FCT\0'
        head = struct.unpack('<III', head_raw[4:])
        [count, list_off, unk_C] = head
        assert unk_C == 0
        assert list_off == 0x10 == file.tell()
        file_list = [struct.unpack('<II', file.read(8)) for _ in range(count)]
        file.seek((file.tell() + 15)//16*16)
        dirpath = fpath + '.dir'
        os.makedirs(dirpath, exist_ok=True)
        for i, (offset, size) in enumerate(file_list):
            assert file.tell() == offset
            data = file.read(size)
            ext = '.'+data[:3].decode()
            destpath = os.path.join(dirpath, str(i) + ext)
            with open(destpath, 'wb') as outfile:
                outfile.write(data)




