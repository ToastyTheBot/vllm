from PIL import Image, ImageDraw
# Deterministic, unambiguous content so a correct answer cannot be a lucky guess.
img = Image.new("RGB", (448, 448), "white")
d = ImageDraw.Draw(img)
d.ellipse([40, 40, 200, 200], fill="red")          # red circle, top-left
d.rectangle([250, 250, 410, 410], fill="blue")     # blue square, bottom-right
img.save("/workspace/shapes.png")

img2 = Image.new("RGB", (448, 448), "white")
d2 = ImageDraw.Draw(img2)
d2.rectangle([60, 60, 388, 388], outline="black", width=8)
try:
    from PIL import ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 240)
except Exception:
    f = None
d2.text((150, 90), "7", fill="black", font=f)
img2.save("/workspace/digit.png")
print("wrote shapes.png (red circle + blue square) and digit.png (the digit 7)")
