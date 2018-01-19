
import cv2
class mjpg_stream():
    buf = []

    def __init__(self):
        self.fd = open('/run/mjpg/input_pipe', 'wb')
      #  self.fd = open('./temp.jpg', 'wb')
        self.buf.extend(['s','t','a','r','t','\n','\0'])
        self.buf.append('\0')
        self.buf.append('\0')
        self.buf.append('\0')
        self.buf.append('\0')
        self.buf.append('\0')
        self.d = 0
    def imout(self,frame):
        file = cv2.imencode(".jpg", frame)
        self.d = self.d + 1
        self.buf[8] = file[1].shape[0] >> 24 & 0xff
        self.buf[9] = file[1].shape[0] >> 16 & 0xff
        self.buf[10] = file[1].shape[0] >> 8 & 0xff
        self.buf[11] = file[1].shape[0]  & 0xff
        self.fd.write(bytearray(self.buf))

        self.fd.flush()
        self.fd.write(file[1])
        self.fd.flush()
        pass
    def __del__(self):
        self.fd.close()

