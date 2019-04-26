import struct
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image
import imageio

FILL_MODES = {
    0x08: -1,
    0x09: -2,
    0x0A: 0, # I think
    0x0B: 0,
    0x0C: 0,
}

BPP_HALF = 4
BPP_FULL = 5

FRAMES_PER_SECOND = 60

class PTXImage:
    fmt = '<BBHHHBBBBIBBBBII'
    assert struct.calcsize(fmt) == 0x1C
    
    def __init__(self, file):
        # Allowing a file pointer.
        #? Use a factory method instead?
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as file:
                self.init_file(file)
                assert not file.read()
        else:
            self.init_file(file)
    
    def init_file(self, file):
        start = file.tell()
        head_raw = file.read(0x20)
        assert head_raw[:4] == b'PTX@'
        head = struct.unpack(self.fmt, head_raw[4:])
        [ # 4 bytes per line:
          t_width, t_height, width_aligned,
          width, height,
          bpp_type, unk_D, unk_E, colors_8, #_D=1, _E=3
          colors_count,
          unk_14, unk_15, unk_16, unk_17,
          palette_off,
          bitmap_off,
        ] = head
        assert bpp_type in (BPP_HALF, BPP_FULL)
        assert (unk_D, unk_E) == (1, 3)
        assert colors_8 == colors_count//8
        if palette_off != 0:
            assert file.tell() == start + palette_off
            file.seek(start + palette_off)
            palette_raw = file.read(colors_count*4)
        else:
            palette_raw = None
        pixel_bytecount = width_aligned * height // (2 if bpp_type == BPP_HALF else 1)
        file.seek(start + bitmap_off)
        bitmap_raw = file.read(pixel_bytecount)
        # assert not file.read()
        
        # palette = palette_raw and list(taken(palette_raw, 4))
        palette = palette_raw and np.array(
                list(palette_raw),
                dtype=np.uint8,
        ).reshape((-1,4))
        if bpp_type == BPP_FULL:
            bitmap = list(bitmap_raw)
            # tiles = list(taken(taken(bitmap, 16), 8))
            tiles = np.array(bitmap).reshape((-1, 8, 16))
        else:
            bitmap = [
                    nybble
                    for byte in bitmap_raw
                    for nybble in [byte%0x10, byte//0x10]]
            # tiles = list(taken(taken(bitmap, 32), 8))
            tiles = np.array(bitmap).reshape((-1, 8, 32))
        self.head_raw = head_raw
        self.head = head
        self.width = width
        self.width_aligned = width_aligned
        self.height = height
        self.fill_mode = unk_15
        self.palette = palette
        self.bitmap = bitmap
        self.tiles = tiles
    
    def full_bitmap(self):
        return list(map(self.palette.__getitem__, self.bitmap))

    def pixels(self):
        tile_w = len(self.tiles[0][0])
        assert self.width_aligned % tile_w == 0
        tiles_per_line = self.width_aligned // tile_w
        #return flatten(flatten(flatten(starmap(zip, taken(self.tiles, tiles_per_line)))))
        return (self.tiles
                .reshape((-1, tiles_per_line, 8, tile_w)) #row, col, tilerow, tilecol
                .transpose((0,2,1,3)) #row, tilerow, col, tilecol
                .reshape((-1,)))
    
    def toImage(self):
        from PIL import Image
        shape = (self.width_aligned, self.height)
        palette = self.palette
        pixels = self.pixels()
        # bitmap = bytes(flatten(map(palette.__getitem__, pixels)))
        # bitmap = bytes(np.take(palette, pixels, 0).flat)
            # #^ Without passing an iterator, each element is encoded as four bytes.
        bitmap = palette[pixels]
        # return Image.frombytes('RGBA', shape, bitmap)
        # return Image.frombytes('RGBA', shape, bytes(bitmap.flat))
        assert bitmap.dtype == np.uint8  #stupid Pillow silently misinterpreting int32s in 3-dimensional shapes as a flattened list of color values.
        return Image.fromarray(bitmap)



class PTGImage:
    fmt = '<IIIHHHHHHHH'
    assert struct.calcsize(fmt) == 0x1C

    #TODO: Make them use a file pointer with a starting offset.
    def __init__(self, file):
        # Allowing a file pointer.
        #? Use a factory method instead?
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as file:
                self.init_file(file)
        else:
            self.init_file(file)

    def init_file(self, file):
        start = file.tell()
        head_raw = file.read(0x20)
        assert head_raw[:4] == b"PTG@"
        head = struct.unpack(self.fmt, head_raw[4:])
        [   # 4 bytes per line (except last):
            unk_04,
            ptx0_off,
            ptx1_off,
            width, height,
            ct0, ct1,
            *unks_18,
        ] = head
        assert unk_04 == 0x20
        assert unks_18 == [0,0,0,0]
        
        pieces = []
        rfmt = '<' + 'H'*10
        rfmt_sz = struct.calcsize(rfmt)
        for _ in range(ct0 + ct1):
            row = struct.unpack(rfmt, file.read(rfmt_sz))
            pieces.append(row)
        file.seek(start + ptx0_off)
        ptx0 = PTXImage(file)
        file.seek(start + ptx1_off)
        ptx1 = PTXImage(file)
        ptx1.palette = ptx0.palette
        
        self.head_raw = head_raw
        self.head = head
        self.cts = ct0, ct1
        self.ptxs = ptx0, ptx1
        self.width = width
        self.height = height
        self.pieces = pieces[:ct0], pieces[ct0:]

    def toImage(self):
        shape = (self.width, self.height)
        palette = self.ptxs[0].palette
        # Ugh fill color index.
        # Canvas needs to round up? Stupid 0609-bust being not rounded.
        # canvas = np.zeros((shape[1]+32, shape[0]+32), dtype=int) + FILL_MODES[self.ptxs[0].fill_mode]
        canvas = np.zeros((shape[1]+32, shape[0]+32), dtype=int) + find_fill(palette)
        # Add 0-color.
        # palette = np.concatenate((self.ptxs[0].palette, [[0,0,0,0]]))
        # canvas = np.zeros(shape[::-1], dtype=int) - 1
        for block, ptx in zip(self.pieces, self.ptxs):
            pixels = ptx.pixels().reshape((-1, ptx.width_aligned))
            for row in block:
                [rx0, ry0, wx0, wy0, unk0,
                 rx1, ry1, wx1, wy1, unk1] = row
                canvas[wy0:wy1,wx0:wx1] = pixels[ry0:ry1,rx0:rx1]
        canvas = canvas[:shape[1], :shape[0]]
        # bitmap = bytes(np.take(palette, canvas, 0).flat)
        # return Image.frombytes('RGBA', shape, bitmap)
        bitmap = np.take(palette, canvas, 0)
        assert bitmap.dtype == np.uint8
        return Image.fromarray(bitmap)


class PTAImage:
    head_fmt = '<III'
    assert struct.calcsize(head_fmt) == 0x0C
    def __init__(self, file):
        # Allowing a file pointer.
        #? Use a factory method instead?
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as file:
                self.init_file(file)
                assert not file.read()
        else:
            self.init_file(file)
    
    def init_file(self, file):
        start = file.tell()
        head_raw = file.read(0x10)
        assert head_raw[:4] == b'PTA\0'
        head = struct.unpack(self.head_fmt, head_raw[4:])
        [
            chunk0_off,
            chunk1_off,
            unk_C,
        ] = head
        assert unk_C == 0
        self.head = head
        
    ## Chunk 0 ##
    
        assert file.tell() == start + chunk0_off
        offs = read_ints(file, 0x2C, signed=True)
        assert offs[-1] == -1
        
        # Table 0: Sprite metadata.
        table0_sz = offs[0] - file.tell()
        table0 = list(struct.iter_unpack('<hhhh', file.read(table0_sz)))
        self.table0 = table0
        
        # Table 1: Animation data.
        table1 = [None]*0x2B #from array of ascending ints.
        for i, beg in enumerate(offs):
            if beg <= 0:
                continue
            file.seek(start+beg)
            rows = []
            row = [None] #dummy placeholder
            while row[0] != 0:
                row = read_ints(file, 4, size=2, signed=False)
                assert row[1] == 0
                assert 0 <= row[2] < 0x100
                assert 0 <= row[3] < 0x100
                rows.append(row)
            table1[i] = rows
        self.table1 = table1
    
    ## Chunk 1 ##
        start1 = start + chunk1_off
        #assert file.tell() == start1
        file.seek(start1)
        head = struct.unpack('<IIIIII', file.read(0x18))
        [
            palette_ptr,
            palette_flags,
            spriteptrs_ptr,
            spriteptrs_count,
            ptxptrs_ptr,
            ptxptrs_count, #maybe
        ] = head
        self.chunk1_head = head
        
        assert file.tell() == start1 + spriteptrs_ptr
        spriteptrs = read_ints(file, spriteptrs_count)
        assert file.tell() == start1 + ptxptrs_ptr
        ptxptrs = read_ints(file, ptxptrs_count)
        filler = file.read(start1 + palette_ptr - file.tell())
        assert not filler or set(filler) == {0x90}
        assert not any(ptr == 0x90909090 for ptr in chain(spriteptrs, ptxptrs))
        
        # Read palettes.
        assert file.tell() == start1 + palette_ptr
        palette_end = spriteptrs[0]
        sz = palette_end - palette_ptr
        palettes_array = np.array(
            list(struct.iter_unpack('<BBBB', file.read(sz))),
            dtype=np.uint8,
        ).reshape((-1, 16, 4))
            #^ Assumes 16-color palettes.
        self.palette_flags = palette_flags
        n = palette_flags
        #palettes = [None] * n.bit_length()
        palettes = np.zeros((16, 16, 4), dtype=np.uint8)
        i = 0
        for palette in palettes_array:
            while n&1 == 0:
                i += 1
                n >>= 1
            palettes[i] = palette
            n >>= 1
            i += 1
        self.palettes_array = palettes_array
        self.palettes = palettes
        assert bin(palette_flags).count('1') == len(palettes_array)
        blanks = set(map(tuple, palettes_array[:,0,:]))
        assert len(blanks) == 1, "Not all blanks are the same!"
        self.blank, = blanks
            # Note that the blank will be a tuple.
        assert self.blank[-1] == 0, "Blank alpha not 0!"
        
        # Read sprite stuff.
        assert file.tell() == start1 + spriteptrs[0]
        spritestuffs = {}
        for ptr in spriteptrs:
            if ptr in spritestuffs:
                continue
            file.seek(start1 + ptr)
            spritestuffs[ptr] = SpriteStuff(file)
        self.spriteptrs = spriteptrs
        self.spritestuffs = spritestuffs

        # Read PTXs.
        ptxs = []
        for ptr in ptxptrs:
            file.seek(start1 + ptr)
            ptxs.append(PTXImage(file))
        self.ptxs = ptxs
    
    def getsprite(self, i):
        """Return the image for the requested sprite.
        """
        sprite = self.spritestuffs[self.spriteptrs[i]]
        palette = self.palettes[sprite.pal]
        fill_color = find_fill(palette)
        shape = (sprite.width, sprite.height)
        canvas = np.full((shape[1]+32, shape[0]+32, 4), fill_color)
        
        for block, ptx_idx, pal_id in zip(sprite.pieces, sprite.ptxs, sprite.pals):
            ptx = self.ptxs[ptx_idx]
            pixels = ptx.pixels().reshape((-1, ptx.width_aligned))
            for row in block:
                [rx0, ry0, wx0, wy0, unk0,
                 rx1, ry1, wx1, wy1, unk1] = row
                #inversions are when e.g. wx0 > wx1:
                wx = handle_inversion(wx0, wx1)
                wy = handle_inversion(wy0, wy1)
                rx = handle_inversion(rx0, rx1)
                ry = handle_inversion(ry0, ry1)
                canvas[wy, wx] = palette[pixels[ry, rx]]
        canvas = canvas[:shape[1], :shape[0]]
        bitmap = canvas
        assert bitmap.dtype == np.uint8
        return Image.fromarray(bitmap)

    # Assume single palette per sprite.
    # Assume color 0 for transparency.
    #? Is this necessary? We'll need to blend multiple palettes together anyway.
    def getspritedata(self, i):
        """The (pixels, palette) for the requested sprite.
        """
        sprite = self.spritestuffs[self.spriteptrs[i]]
        palette = self.palettes[sprite.pal]
        shape = (sprite.width, sprite.height)
        canvas = np.zeros((shape[1]+32, shape[0]+32), dtype=int)
        
        for block, ptx_idx in zip(sprite.pieces, sprite.ptxs):
            ptx = self.ptxs[ptx_idx]
            pixels = ptx.pixels().reshape((-1, ptx.width_aligned))
            for row in block:
                [rx0, ry0, wx0, wy0, unk0,
                 rx1, ry1, wx1, wy1, unk1] = row
                #inversions are when e.g. wx0 > wx1:
                wx = handle_inversion(wx0, wx1)
                wy = handle_inversion(wy0, wy1)
                rx = handle_inversion(rx0, rx1)
                ry = handle_inversion(ry0, ry1)
                canvas[wy, wx] = pixels[ry, rx]
        canvas1 = np.array(canvas[:shape[1], :shape[0]])
        return canvas1, palette

    def getshiftsprite(self, offset):
        """Get the shifted sprite represented by given offset of file.
        """
        assert offset % 8 == 0
        index = (offset - 0xC0)//8
        shiftdata = self.table0[index]
        [
          i,
          x_off,
          y_off,
          unk_6,
        ] = shiftdata
        assert unk_6 in (0, 0x10)
        #? What's a good way to shift an image?
        #... Let's not worry about it yet.
        canvas, palette = self.getspritedata(i)
        canvas1 = np.roll(canvas, shift=(y_off, x_off), axis=(0, 1))
            # Definitely y then x.
            # Definitely positive x.
            # Definitely positive y.
        bitmap = palette[canvas1]
        assert bitmap.dtype == np.uint8
        return Image.fromarray(bitmap)

    def getshiftsprite2(self, offset):
        """Get the shifted sprite represented by given offset of file. Shift in an enlarged canvas.
        """
        assert offset % 8 == 0
        index = (offset - 0xC0)//8
        shiftdata = self.table0[index]
        [
          i,
          x_off,
          y_off,
          unk_6,
        ] = shiftdata
        assert unk_6 in (0, 0x10)
        #? What's a good way to shift an image?
        #... Let's not worry about it yet.
        canvas, palette = self.getspritedata(i)
        height, width = canvas.shape
        canvas1 = np.zeros((height*2, width*2), dtype=np.uint8)
        canvas1[height//2:-height//2, width//2:-width//2] = canvas
        canvas1 = np.roll(canvas1, shift=(y_off, x_off), axis=(0, 1))
            # Definitely y then x.
            # Definitely positive x.
            # Definitely positive y.
        bitmap = palette[canvas1]
        assert bitmap.dtype == np.uint8
        return Image.fromarray(bitmap)


    def getcombinedsprite(self, offset, count):
        #WTF COUNT CAN BE 0 (Necromancer disappearance).
        sprites = [self.getshiftsprite(offset + 8*i) for i in range(count)]
        # Now combine them.
        canvas = Image.new('RGBA', sprites[0].size, self.blank)
        for sprite in sprites:
            canvas.paste(sprite, mask=sprite)
        return canvas

    def getcombinedsprite2(self, offset, count):
        #WTF COUNT CAN BE 0 (Necromancer disappearance).
        sprites = [self.getshiftsprite2(offset + 8*i) for i in range(count)]
        # Now combine them.
        canvas = Image.new('RGBA', sprites[0].size, self.blank)
        for sprite in sprites:
            canvas.paste(sprite, mask=sprite)
        return canvas

    # For debugging.
    def getuncombinedsprite(self, offset, count):
        sprites = [self.getshiftsprite(offset + 8*i) for i in range(count)]
        return sprites
    
    def getanimation(self, index):
        """Returns (images, durframes, loopi or None) or None.
        
        If an animation is None, that means this unit doesn't have an animation at that index.
        
        `loopi` is the index at which to restart the loop. If a `loopi` is missing, this means the animation doesn't loop.
        
        Duration is in frames, or 1/60ths of a second.
        """
        anim = self.table1[index]
        if anim is None:
            return None
        *rows, [zero0, zero1, loopflag, loopi] = anim
        assert zero0 == zero1 == 0
        assert (loopi == 0xFF) == (loopflag == 0xFF)
        assert (loopi != 0xFF) == (loopflag == 0)
        sprites = []
        durations = []
        if loopi == 0xFF:
            loopi = None
        for j, (off, zero, count, dur) in enumerate(rows):
            assert zero == 0
            # Handle count=0.
            if count == 0:  #Blank image.
                assert j != 0, "First image is blank!"
                # Use the previous image's dimensions.
                #? Can I just use a 0-pixel image?
                    # No, I want the right transparency.
                img = Image.new('RGBA', img.size, self.blank)
            else:
                img = self.getcombinedsprite2(off, count)
            sprites.append(img)
            durations.append(dur)
        return sprites, durations, loopi


def handle_inversion(x0, x1):
    if x0 < x1:
        return slice(x0, x1)
    elif x1 == 0:  #Sending -1 will result in the wrong thing.
        return slice(x0-1, None, -1)
    else:
        return slice(x0-1, x1-1, -1)


class SpriteStuff:
    '''Represents a single sprite.
    
    .pieces: A list of extractions from the PTX.
        Same format as PTG's pieces.
    '''
    def __init__(self, file):
        pieces = []
        ptxs = []
        pals = []
        width = None
        height = None
        count = read_int(file, 2)
        while count:
            head = struct.unpack('<HHHH', file.read(0x8))
            [
                ptx_id,
                pal,
                width_i, #maybe
                height_i, #maybe
            ] = head
            if width is None:
                width = width_i
                height = height_i
            else:
                assert (width, height) == (width_i, height_i)
            
            block = list(struct.iter_unpack('<' + 'h'*(0x14//2), file.read(count*0x14)))
            assert all(row[4] == row[9] == 0 for row in block)
            
            ptxs.append(ptx_id)
            pieces.append(block)
            pals.append(pal)
            
            count = read_int(file, 2)
        
        self.head = head
        self.width = width
        self.height = height
        self.pals = pals
        assert len(set(pals)) == 1
        self.pal = pals[0]
        self.pieces = pieces
        self.ptxs = ptxs


def read_ints(file, n, size=4, signed=False):
    """Read ints from given file.
    """
    return [int.from_bytes(file.read(size), 'little', signed=signed) for _ in range(n)]

def read_int(file, size=4, signed=False):
    return int.from_bytes(file.read(size), 'little', signed=signed)

def flatten(itr):
    for xs in itr:
        yield from xs

def taken(itr, n):
    return zip(*n*[iter(itr)])


def find_fill(palette):
    """Determines the fill color's index.
    """
    # Find first color which has 0 transparency.
    transparents = [i for i, color in enumerate(palette) if color[-1] == 0]
    assert len(set(transparents)) == 1
    return transparents[0]

