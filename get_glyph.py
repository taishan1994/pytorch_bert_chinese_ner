from PIL import Image, ImageFont, ImageDraw
import numpy as np
 
 
def CreateImg(text, fontPath):
    fontSize = 24
    liens = text.split('\n')
    # 画布大小为24×24，颜色为黑色
    im = Image.new("RGB", (24, 24), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    #字体样式，文章结尾我会放上连接
    
    font = ImageFont.truetype(fontPath, fontSize)
    # 文字颜色
    dr.text((0, 0), text, font=font, fill="#FFFFFF")
    im.save('output.png')
    im.show()

    data = Image.open("output.png")
    data = data.convert("L")
    data = np.array(data).tolist()
    # print(data)
    return data 

def is_Chinese(word):
  for ch in word:
    if '\u4e00' <= ch <= '\u9fff':
        return True
  return False

vocab_path = "model_hub/chinese-bert-wwm-ext/vocab.txt"
with open(vocab_path, "r") as fp:
  vocab = fp.read().strip().split("\n")

fontPath = ['data/resume/' + font for font in ["方正古隶繁体.ttf", "STFangsong.ttf", "STXingkai.ttf"]]

for fontp in fontPath:
  res = []
  for word in vocab:
    if len(word) == 1 and is_Chinese(word):
      tmp = CreateImg(word, fontp)
    else:
      tmp = [[0 for _ in range(24)] for _ in range(24)]
    res.append(tmp)
  res = np.array(res)
  np.save(fontp + ".npy", res)