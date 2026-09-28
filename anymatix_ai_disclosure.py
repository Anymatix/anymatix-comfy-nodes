"""
The label every generated result carries: "this was made by AI".

WHY IT EXISTS

  EU AI Act art. 50(2), in force since 2 August 2026: a provider of a system
  that generates image, audio or video must mark the output in a
  machine-readable way. Anymatix decided (2026-09-28, TRACKERS item
  `c2pa-content-credentials-marking-decided-2026-08-13-not`) to meet it with
  UNSIGNED standard metadata: no certificate, no C2PA manifest, no watermark.
  A signing key inside a desktop app can be extracted, so a signature would
  change the effort needed to forge the label, not whether it can be forged.

WHAT IS WRITTEN -- and it is the same three constants in every format

  * machine-readable: XMP `Iptc4xmpExt:DigitalSourceType` =
    `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`
    (the IPTC term the AI Act guidance and C2PA both use);
  * human-readable: `Created by Anymatix using AI generators`;
  * software: `Anymatix`. No version: the node pack does not know the app's
    version, and inventing one would make the label say something untrue.

  NEVER the prompt, the user, a path, a machine name or any identifier. The
  label says THAT the content is AI-generated, never what was asked or who
  asked. That is what keeps it disclosure and not surveillance, and why it
  stays on in confidential mode. Every byte written here is a module constant:
  nothing a caller passes can reach the file, by construction.

WHERE IT IS WRITTEN

  On the machine that runs the job, by the node that writes the result --
  local, RunPod or SSH alike. Not in the Electron renderer, which only ever
  sees a downloaded copy.

  The savers do not share one choke point (`write_image` publishes through
  `anymatix_atomic_write.publish`, the audio saver through `atomic_output`,
  the video saver through its own inline replace), so each calls THIS module
  once, on its staged temp file or buffer, before the atomic publish. The
  per-format knowledge lives here and nowhere else.

  | extension          | mechanism                                             |
  |--------------------|-------------------------------------------------------|
  | png                | iTXt `XML:com.adobe.xmp` + tEXt Software, Description |
  | jpg, jpeg          | APP1 XMP segment + COM comment                        |
  | webp               | VP8X header with the XMP flag + `XMP ` chunk          |
  | gif                | XMP application extension + comment extension         |
  | tiff               | IFD0 tags 270 ImageDescription, 305 Software, 700 XMP |
  | exr                | header string attributes (comments, software,         |
  |                    | digitalSourceType) -- EXR has no XMP convention       |
  | avif               | HEIF `mime` item (application/rdf+xml) linked `cdsc`  |
  |                    | to the primary image                                  |
  | mp4, mov, m4a      | top-level XMP `uuid` box, plus the encoder's comment  |
  | mp3                | ID3v2 PRIV `XMP` frame, plus the encoder's COMM/TXXX  |
  | wav                | RIFF `_PMX` XMP chunk, plus the encoder's INFO ICMT   |
  | mkv, flac          | the encoder's own tags only (`CONTAINER_TAGS`):       |
  |                    | neither container has an XMP convention               |
  | bmp                | NOT MARKABLE: the format has no metadata field        |
  | json (sidecars)    | EXEMPT: not media, art. 50(2) does not apply          |

WHAT IS NOT NEGOTIABLE HERE

  The picture is the product and the label is best-effort on top of it. A
  failure to label NEVER fails the save and NEVER corrupts the file: every
  marker computes the labelled bytes in full before anything is written,
  a structure it does not recognise makes it return None (the file is left
  exactly as the encoder wrote it), and the in-place writes are staged and
  `os.replace()`d, or -- for the one append-only case, video -- truncated back
  on any error.

  Nothing here claims the label cannot be removed. It can: one re-encode.

This module imports neither `comfy` nor `folder_paths`, so it is tested
outside ComfyUI: `tests/test_ai_disclosure.py`.
"""

import os
import struct
import zlib

try:
    from .anymatix_atomic_write import cleanup_temp, temp_path_for
except ImportError:
    # Loaded standalone (tests via spec_from_file_location), with no package
    # context for a relative import to resolve against.
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from anymatix_atomic_write import cleanup_temp, temp_path_for


DIGITAL_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)
DESCRIPTION = "Created by Anymatix using AI generators"
SOFTWARE = "Anymatix"

_DST = DIGITAL_SOURCE_TYPE.encode("ascii")
_DESCRIPTION = DESCRIPTION.encode("ascii")
_SOFTWARE = SOFTWARE.encode("ascii")

#: The whole XMP packet. A constant: nothing variable ever enters it. The
#: `begin` attribute is the UTF-8 byte-order mark, as the XMP spec requires.
XMP_PACKET = b"".join(
    [
        b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n',
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/">\n',
        b' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n',
        b'  <rdf:Description rdf:about=""\n',
        b'    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"\n',
        b'    xmlns:dc="http://purl.org/dc/elements/1.1/"\n',
        b'    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n',
        b'    Iptc4xmpExt:DigitalSourceType="' + _DST + b'"\n',
        b'    xmp:CreatorTool="' + _SOFTWARE + b'">\n',
        b"   <dc:description>\n",
        b"    <rdf:Alt>\n",
        b'     <rdf:li xml:lang="x-default">' + _DESCRIPTION + b"</rdf:li>\n",
        b"    </rdf:Alt>\n",
        b"   </dc:description>\n",
        b"  </rdf:Description>\n",
        b" </rdf:RDF>\n",
        b"</x:xmpmeta>\n",
        b'<?xpacket end="w"?>',
    ]
)

#: Key/value tags for the encoder itself (ffmpeg `-metadata`, PyAV
#: `container.metadata`). Each muxer keeps the keys it has a field for and
#: drops the rest in silence -- never an error: `comment` becomes MP4/MOV
#: `(c)cmt`, ID3 COMM, RIFF ICMT, a Matroska or Vorbis COMMENT;
#: `DIGITAL_SOURCE_TYPE` survives in Matroska, FLAC and ID3 (TXXX), which is
#: the machine-readable label for the two containers that have no XMP.
CONTAINER_TAGS = (
    ("comment", DESCRIPTION),
    ("DIGITAL_SOURCE_TYPE", DIGITAL_SOURCE_TYPE),
)


def ffmpeg_metadata_args():
    """`CONTAINER_TAGS` as ffmpeg output options."""
    args = []
    for key, value in CONTAINER_TAGS:
        args += ["-metadata", f"{key}={value}"]
    return args


def set_container_tags(container):
    """`CONTAINER_TAGS` onto a PyAV output container, before the first mux."""
    for key, value in CONTAINER_TAGS:
        container.metadata[key] = value


# ------------------------------------------------------------ the formats ----


def _png_chunk(kind, body):
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def _mark_png(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    after_ihdr = 8 + 12 + struct.unpack(">I", data[8:12])[0]
    # iTXt: keyword NUL, compression flag, method, language NUL, translated NUL.
    chunks = (
        _png_chunk(b"iTXt", b"XML:com.adobe.xmp" + b"\x00" * 5 + XMP_PACKET)
        + _png_chunk(b"tEXt", b"Software\x00" + _SOFTWARE)
        + _png_chunk(b"tEXt", b"Description\x00" + _DESCRIPTION)
    )
    # Straight after IHDR: before IDAT, where the XMP spec asks for it.
    return data[:after_ihdr] + chunks + data[after_ihdr:]


def _jpeg_segment(marker, payload):
    return b"\xff" + marker + struct.pack(">H", len(payload) + 2) + payload


def _mark_jpeg(data):
    if data[:2] != b"\xff\xd8":
        return None
    at = 2
    # Past the leading APPn run: JFIF's APP0 must stay first after SOI.
    while data[at] == 0xFF and 0xE0 <= data[at + 1] <= 0xEF:
        at += 2 + struct.unpack(">H", data[at + 2 : at + 4])[0]
    segments = _jpeg_segment(
        b"\xe1", b"http://ns.adobe.com/xap/1.0/\x00" + XMP_PACKET
    ) + _jpeg_segment(b"\xfe", _DESCRIPTION)
    return data[:at] + segments + data[at:]


def _riff_chunk(kind, body):
    pad = b"\x00" if len(body) % 2 else b""
    return kind + struct.pack("<I", len(body)) + body + pad


_WEBP_XMP_FLAG = 0x04
_WEBP_ALPHA_FLAG = 0x10


def _webp_canvas(data):
    """Width, height and alpha of a simple (non-VP8X) WebP, from its bitstream."""
    kind = data[12:16]
    if kind == b"VP8L":
        if data[20] != 0x2F:
            return None
        bits = struct.unpack("<I", data[21:25])[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        alpha = bool((bits >> 28) & 1)
        return width, height, alpha
    if kind == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height, False
    return None


def _mark_webp(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    if 8 + struct.unpack("<I", data[4:8])[0] != len(data):
        return None
    kind = data[12:16]
    if kind == b"VP8X":
        chunks = data[12:20] + bytes([data[20] | _WEBP_XMP_FLAG]) + data[21:]
    elif kind in (b"VP8 ", b"VP8L"):
        canvas = _webp_canvas(data)
        if canvas is None:
            return None
        width, height, alpha = canvas
        flags = _WEBP_XMP_FLAG | (_WEBP_ALPHA_FLAG if alpha else 0)
        vp8x = _riff_chunk(
            b"VP8X",
            bytes([flags, 0, 0, 0])
            + (width - 1).to_bytes(3, "little")
            + (height - 1).to_bytes(3, "little"),
        )
        chunks = vp8x + data[12:]
    else:
        return None
    # The XMP chunk goes last: the container spec orders it after the image.
    chunks += _riff_chunk(b"XMP ", XMP_PACKET)
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WEBP" + chunks


#: The XMP spec's GIF trailer: whatever byte a decoder lands on while it reads
#: the raw packet as sub-blocks, the lengths walk it down to the terminator.
_GIF_XMP_TRAILER = b"\x01" + bytes(range(255, -1, -1)) + b"\x00"


def _mark_gif(data):
    if data[:6] not in (b"GIF87a", b"GIF89a") or b"\x00" in XMP_PACKET:
        return None
    packed = data[10]
    at = 13
    if packed & 0x80:
        at += 3 * (2 << (packed & 0x07))
    comment = b"\x21\xfe" + bytes([len(_DESCRIPTION)]) + _DESCRIPTION + b"\x00"
    xmp = b"\x21\xff\x0bXMP DataXMP" + XMP_PACKET + _GIF_XMP_TRAILER
    # Extensions are a GIF89a feature; 89a reads every 87a file.
    return b"GIF89a" + data[6:at] + comment + xmp + data[at:]


_TIFF_ASCII = 2
_TIFF_BYTE = 1


def _mark_tiff(data):
    """
    A new IFD0 appended at the end, the header pointed at it.

    Nothing the image already holds moves, so every strip offset stays valid;
    the old IFD0 is left in place, unreferenced. Classic TIFF only: BigTIFF
    (version 43) is returned unmarked rather than guessed at.
    """
    if data[:2] == b"II":
        order = "<"
    elif data[:2] == b"MM":
        order = ">"
    else:
        return None
    if struct.unpack(order + "H", data[2:4])[0] != 42:
        return None
    ifd = struct.unpack(order + "I", data[4:8])[0]
    count = struct.unpack(order + "H", data[ifd : ifd + 2])[0]
    entries = {}
    for index in range(count):
        raw = data[ifd + 2 + 12 * index : ifd + 14 + 12 * index]
        entries[struct.unpack(order + "H", raw[:2])[0]] = raw
    next_ifd = data[ifd + 2 + 12 * count : ifd + 6 + 12 * count]

    out = bytearray(data)
    if len(out) % 2:
        out += b"\x00"

    def entry(tag, kind, payload):
        if len(payload) <= 4:
            return struct.pack(order + "HHI", tag, kind, len(payload)) + payload.ljust(
                4, b"\x00"
            )
        offset = len(out)
        out.extend(payload)
        if len(out) % 2:
            out.append(0)
        return struct.pack(order + "HHII", tag, kind, len(payload), offset)

    entries[270] = entry(270, _TIFF_ASCII, _DESCRIPTION + b"\x00")
    entries[305] = entry(305, _TIFF_ASCII, _SOFTWARE + b"\x00")
    entries[700] = entry(700, _TIFF_BYTE, XMP_PACKET)
    new_ifd = len(out)
    out += struct.pack(order + "H", len(entries))
    out += b"".join(entries[tag] for tag in sorted(entries)) + next_ifd
    if len(out) >= 1 << 32:
        return None
    out[4:8] = struct.pack(order + "I", new_ifd)
    return bytes(out)


#: Scan lines per chunk, by OpenEXR compression code.
_EXR_LINES_PER_CHUNK = {0: 1, 1: 1, 2: 1, 3: 16, 4: 32, 5: 16, 6: 32, 7: 32, 8: 32, 9: 256}
_EXR_TILED, _EXR_NON_IMAGE, _EXR_MULTIPART = 0x200, 0x800, 0x1000


def _exr_string(name, value):
    return name + b"\x00string\x00" + struct.pack("<i", len(value)) + value


def _mark_exr(data):
    """
    Three string attributes appended to the header, the offset table shifted.

    OpenEXR has no XMP convention; `comments` is a standard attribute, and
    `software` and `digitalSourceType` are ordinary string attributes every
    reader keeps. Single-part scan-line files only -- the only kind the image
    node writes -- anything else is returned unmarked.
    """
    if data[:4] != b"\x76\x2f\x31\x01":
        return None
    flags = struct.unpack("<I", data[4:8])[0]
    if flags & (_EXR_TILED | _EXR_NON_IMAGE | _EXR_MULTIPART):
        return None
    names = set()
    compression = None
    window = None
    at = 8
    while data[at] != 0:
        name_end = data.index(b"\x00", at)
        type_end = data.index(b"\x00", name_end + 1)
        size = struct.unpack("<i", data[type_end + 1 : type_end + 5])[0]
        value = data[type_end + 5 : type_end + 5 + size]
        name = data[at:name_end]
        names.add(name)
        if name == b"compression":
            compression = value[0]
        elif name == b"dataWindow":
            window = struct.unpack("<4i", value)
        at = type_end + 5 + size
    header_end = at
    if compression not in _EXR_LINES_PER_CHUNK or window is None:
        return None
    lines = _EXR_LINES_PER_CHUNK[compression]
    chunks = -(-(window[3] - window[1] + 1) // lines)
    table = header_end + 1
    table_end = table + 8 * chunks
    offsets = struct.unpack(f"<{chunks}Q", data[table:table_end])
    if min(offsets) != table_end or max(offsets) >= len(data):
        return None

    added = b"".join(
        _exr_string(name, value)
        for name, value in (
            (b"comments", _DESCRIPTION),
            (b"software", _SOFTWARE),
            (b"digitalSourceType", _DST),
        )
        if name not in names
    )
    shifted = struct.pack(f"<{chunks}Q", *(offset + len(added) for offset in offsets))
    return data[:header_end] + added + data[header_end:table] + shifted + data[table_end:]


# ----------------------------------------------------- ISO base media file ----

#: Adobe's UUID for an XMP box (XMP spec part 3, MPEG-4).
_XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")


def _box(kind, body):
    return struct.pack(">I", 8 + len(body)) + kind + body


def _xmp_uuid_box():
    return _box(b"uuid", _XMP_UUID + XMP_PACKET)


def _top_level_boxes(read_at, length):
    """
    [(type, start, size)] of the top-level boxes, or None when they do not
    tile the file exactly -- in which case nothing may be appended.
    """
    boxes = []
    at = 0
    while at < length:
        header = read_at(at, 16)
        if len(header) < 8:
            return None
        size, kind = struct.unpack(">I4s", header[:8])
        if size == 1:
            if len(header) < 16:
                return None
            size = struct.unpack(">Q", header[8:16])[0]
        elif size == 0:
            size = length - at
        if size < 8 or at + size > length:
            return None
        boxes.append((kind, at, size))
        at += size
    return boxes


def _mark_isobmff(data):
    """MP4 / MOV / M4A: a top-level XMP `uuid` box appended at the end."""
    boxes = _top_level_boxes(lambda at, n: data[at : at + n], len(data))
    if not boxes or boxes[0][0] != b"ftyp":
        return None
    return data + _xmp_uuid_box()


def _append_isobmff(path):
    """
    `_mark_isobmff` for a file too large to hold in memory (a ProRes master).

    An append moves nothing, so every chunk offset in `moov` stays valid. On
    any error the file is truncated back to the length the encoder left.
    """
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        length = f.tell()

        def read_at(at, n):
            f.seek(at)
            return f.read(n)

        boxes = _top_level_boxes(read_at, length)
        if not boxes or boxes[0][0] != b"ftyp":
            return False
        try:
            f.seek(length)
            f.write(_xmp_uuid_box())
            f.flush()
        except BaseException:
            f.truncate(length)
            raise
    return True


def _full_box(data, start, size):
    """(version, flags, payload) of a FullBox at `start`."""
    version = data[start + 8]
    flags = int.from_bytes(data[start + 9 : start + 12], "big")
    return version, flags, data[start + 12 : start + size]


def _uint(data, at, size):
    return int.from_bytes(data[at : at + size], "big"), at + size


def _mark_heif(data):
    """
    AVIF: the XMP packet as a `mime` item, linked `cdsc` to the primary image.

    That is where HEIF keeps XMP (ISO/IEC 23008-12). The packet's bytes go in
    a new `mdat` placed straight after `meta`, and every existing file-offset
    extent that lies after `meta` is shifted by exactly what was inserted.

    Straight after `meta`, not at the end of the file: measured 2026-09-28,
    OpenCV 4.13 (libavif 1.3.0) refuses to decode an AVIF whose XMP item lies
    after the image data, and decodes the same bytes with the XMP first --
    the order Pillow's own AVIF writer uses.
    """
    boxes = _top_level_boxes(lambda at, n: data[at : at + n], len(data))
    if not boxes or boxes[0][0] != b"ftyp":
        return None
    metas = [box for box in boxes if box[0] == b"meta"]
    if len(metas) != 1:
        return None
    _, meta_start, meta_size = metas[0]
    meta_end = meta_start + meta_size
    if struct.unpack(">I", data[meta_start : meta_start + 4])[0] != meta_size:
        return None  # a 64-bit size: not what any AVIF encoder writes

    children = []
    at = meta_start + 12
    while at < meta_end:
        size, kind = struct.unpack(">I4s", data[at : at + 8])
        if size < 8 or at + size > meta_end:
            return None
        children.append([kind, data[at : at + size]])
        at += size
    found = {kind: body for kind, body in children}
    if not all(kind in found for kind in (b"pitm", b"iloc", b"iinf")):
        return None

    # The primary item.
    version, _, payload = _full_box(found[b"pitm"], 0, len(found[b"pitm"]))
    primary = int.from_bytes(payload[: 2 if version == 0 else 4], "big")

    # iinf: every item ID, and room for one more.
    version, flags, payload = _full_box(found[b"iinf"], 0, len(found[b"iinf"]))
    count_size = 2 if version == 0 else 4
    count = int.from_bytes(payload[:count_size], "big")
    infes = payload[count_size:]
    ids = []
    at = 0
    while at < len(infes):
        size = struct.unpack(">I", infes[at : at + 4])[0]
        if infes[at + 4 : at + 8] != b"infe" or size < 12:
            return None
        infe_version = infes[at + 8]
        if infe_version < 2:
            return None
        id_size = 2 if infe_version == 2 else 4
        ids.append(int.from_bytes(infes[at + 12 : at + 12 + id_size], "big"))
        at += size
    if len(ids) != count:
        return None
    item = max(ids) + 1
    if item > 0xFFFF:
        return None
    infe = _box(
        b"infe",
        b"\x02\x00\x00\x00"
        + struct.pack(">HH", item, 0)
        + b"mime"
        + b"\x00"
        + b"application/rdf+xml\x00",
    )
    new_iinf = _box(
        b"iinf",
        bytes([version])
        + flags.to_bytes(3, "big")
        + (count + 1).to_bytes(count_size, "big")
        + infes
        + infe,
    )

    # iref: "item describes the primary image".
    cdsc = _box(b"cdsc", struct.pack(">HHH", item, 1, primary))
    if b"iref" in found:
        iref_version = found[b"iref"][8]
        if iref_version != 0:
            return None
        new_iref = _box(b"iref", found[b"iref"][8:] + cdsc)
    else:
        new_iref = _box(b"iref", b"\x00\x00\x00\x00" + cdsc)

    # iloc: parse, grow by one entry, then shift.
    iloc = found[b"iloc"]
    version, flags, payload = _full_box(iloc, 0, len(iloc))
    if version > 2:
        return None
    offset_size = payload[0] >> 4
    length_size = payload[0] & 0x0F
    base_size = payload[1] >> 4
    index_size = (payload[1] & 0x0F) if version in (1, 2) else 0
    if offset_size not in (4, 8) or length_size not in (4, 8):
        return None
    id_size = 4 if version == 2 else 2
    at = 2
    item_count, at = _uint(payload, at, 2 if version < 2 else 4)
    entries = []
    for _ in range(item_count):
        item_id, at = _uint(payload, at, id_size)
        method = 0
        if version in (1, 2):
            method, at = _uint(payload, at, 2)
            method &= 0x0F
        reference, at = _uint(payload, at, 2)
        base, at = _uint(payload, at, base_size)
        extent_count, at = _uint(payload, at, 2)
        extents = []
        for _ in range(extent_count):
            index, at = _uint(payload, at, index_size)
            offset, at = _uint(payload, at, offset_size)
            length, at = _uint(payload, at, length_size)
            extents.append([index, offset, length])
        entries.append([item_id, method, reference, base, extents])
    if at != len(payload):
        return None

    def encode_iloc(entries):
        body = bytearray(payload[:2])
        body += len(entries).to_bytes(2 if version < 2 else 4, "big")
        for item_id, method, reference, base, extents in entries:
            body += item_id.to_bytes(id_size, "big")
            if version in (1, 2):
                body += method.to_bytes(2, "big")
            body += reference.to_bytes(2, "big")
            body += base.to_bytes(base_size, "big") if base_size else b""
            body += len(extents).to_bytes(2, "big")
            for index, offset, length in extents:
                body += index.to_bytes(index_size, "big") if index_size else b""
                body += offset.to_bytes(offset_size, "big")
                body += length.to_bytes(length_size, "big") if length_size else b""
        return _box(b"iloc", bytes([version]) + flags.to_bytes(3, "big") + bytes(body))

    xmp_entry = [item, 0, 0, 0, [[0, 0, len(XMP_PACKET)]]]
    probe_iloc = encode_iloc(entries + [xmp_entry])

    old_meta_children = sum(len(body) for _, body in children)
    new_children = []
    for kind, body in children:
        if kind == b"iinf":
            new_children.append([kind, new_iinf])
            if b"iref" not in found:
                new_children.append([b"iref", new_iref])
        elif kind == b"iref":
            new_children.append([kind, new_iref])
        elif kind == b"iloc":
            new_children.append([kind, probe_iloc])
        else:
            new_children.append([kind, body])
    delta = sum(len(body) for _, body in new_children) - old_meta_children

    # Shift every file-offset extent that points past the old meta box. An
    # offset that no longer fits its field raises OverflowError in
    # `encode_iloc`, and the file is left unmarked.
    xmp_mdat = _box(b"mdat", XMP_PACKET)
    shift = delta + len(xmp_mdat)
    for entry in entries:
        _, method, reference, base, extents = entry
        if method != 0 or reference != 0:
            continue
        if base_size and base >= meta_end:
            entry[3] = base + shift
            continue
        for extent in extents:
            if base + extent[1] >= meta_end:
                extent[1] += shift
    xmp_entry[4][0][1] = meta_end + delta + 8
    final_iloc = encode_iloc(entries + [xmp_entry])
    if len(final_iloc) != len(probe_iloc):
        return None
    new_children = [
        [kind, final_iloc if kind == b"iloc" else body] for kind, body in new_children
    ]

    meta_body = data[meta_start + 8 : meta_start + 12] + b"".join(
        body for _, body in new_children
    )
    new_meta = _box(b"meta", meta_body)
    if len(new_meta) - meta_size != delta:
        return None
    return data[:meta_start] + new_meta + xmp_mdat + data[meta_end:]


# ------------------------------------------------------------------ audio ----


def _synchsafe(value):
    return bytes(
        [(value >> 21) & 0x7F, (value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F]
    )


def _unsynchsafe(raw):
    return (raw[0] << 21) | (raw[1] << 14) | (raw[2] << 7) | raw[3]


def _mark_id3(data):
    """MP3: an ID3v2 PRIV frame owned by `XMP`, the convention Adobe writes."""
    body = b"XMP\x00" + XMP_PACKET
    if data[:3] != b"ID3":
        frame = b"PRIV" + _synchsafe(len(body)) + b"\x00\x00" + body
        return b"ID3\x04\x00\x00" + _synchsafe(len(frame)) + frame + data
    major, flags = data[3], data[5]
    # Unsynchronised, extended-header or footed tags are left alone rather
    # than half-understood. ffmpeg writes none of the three.
    if major not in (3, 4) or flags & 0xF0:
        return None
    size = _unsynchsafe(data[6:10])
    frame_size = _synchsafe(len(body)) if major == 4 else struct.pack(">I", len(body))
    frame = b"PRIV" + frame_size + b"\x00\x00" + body
    header = data[:6] + _synchsafe(size + len(frame))
    return header + frame + data[10:]


def _mark_wav(data):
    """WAV: the XMP packet in a `_PMX` chunk, the convention Adobe writes."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    if 8 + struct.unpack("<I", data[4:8])[0] != len(data):
        return None
    chunks = data[12:] + _riff_chunk(b"_PMX", XMP_PACKET)
    if 4 + len(chunks) >= 1 << 32:
        return None
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


# --------------------------------------------------------------- dispatch ----

#: extension -> how its bytes are labelled. An extension that is not here is
#: labelled some other way or not at all, and says which:
#:   mkv, flac -- by the encoder's own tags (`CONTAINER_TAGS`): no XMP exists
#:                for either container;
#:   bmp       -- NOT MARKABLE: the format has no field for metadata;
#:   json      -- EXEMPT: a sidecar, not media.
_MARKERS = {
    "png": _mark_png,
    "jpg": _mark_jpeg,
    "jpeg": _mark_jpeg,
    "webp": _mark_webp,
    "gif": _mark_gif,
    "tif": _mark_tiff,
    "tiff": _mark_tiff,
    "exr": _mark_exr,
    "avif": _mark_heif,
    "mp4": _mark_isobmff,
    "mov": _mark_isobmff,
    "m4a": _mark_isobmff,
    "mp3": _mark_id3,
    "wav": _mark_wav,
}

#: Video can be gigabytes: labelled by appending, never by rewriting.
_APPEND_ONLY = {"mp4", "mov", "m4a"}


def _extension(extension):
    return extension.lower().lstrip(".")


def _not_labelled(what, error):
    print(f"anymatix: AI-disclosure label not written to a {what}: {error!r}")


def mark_bytes(data, extension):
    """
    `data` with the label added, or `data` itself when the format cannot
    carry it, the structure is not one this module recognises, or anything at
    all goes wrong. Never raises.
    """
    marker = _MARKERS.get(_extension(extension))
    if marker is None:
        return data
    try:
        marked = marker(bytes(data))
    except Exception as error:  # the label is best-effort; the output is not
        _not_labelled(_extension(extension), error)
        return data
    return data if marked is None else marked


def mark_file(path, extension):
    """
    Label the file at `path` in place. True when it now carries the label.

    Meant for a staged temp file, before the caller publishes it. The file is
    either labelled completely or left exactly as it was: the labelled bytes
    are written to a sibling and `os.replace()`d over `path`, and video is
    appended to and truncated back on failure. Never raises.
    """
    extension = _extension(extension)
    marker = _MARKERS.get(extension)
    if marker is None:
        return False
    staged = None
    try:
        if extension in _APPEND_ONLY:
            return _append_isobmff(path)
        with open(path, "rb") as f:
            data = f.read()
        marked = marker(data)
        if marked is None:
            return False
        staged = temp_path_for(path)
        with open(staged, "wb") as f:
            f.write(marked)
        os.replace(staged, path)
        staged = None
        return True
    except Exception as error:  # the label is best-effort; the output is not
        _not_labelled(extension, error)
        return False
    finally:
        if staged is not None:
            cleanup_temp(staged)
