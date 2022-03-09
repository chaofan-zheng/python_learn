https://www.jianshu.com/p/694b8639f42f
```python
import msgpack
msgpack.packb([1, 2, 3], use_bin_type=True)
msgpack.unpackb(_, raw=False)
#---------------------------------------------------------
'\x93\x01\x02\x03'
[1, 2, 3]

# 借助于bytesIO打包数据：
import msgpack
from io import BytesIO

buf = BytesIO()
for i in range(100):
   buf.write(msgpack.packb(i, use_bin_type=True))

buf.seek(0)
unpacker = msgpack.Unpacker(buf, raw=False)
for unpacked in unpacker:
    print(unpacked)
```
```python
import msgpack
a = msgpack.dumps([1, 2, 3], use_bin_typ
e=True)

msgpack.loads(a, raw=False)
```
