import time, sys
width = 20
pos = 0
direction = 1
for _ in range(20):
    bar = [" "] * width
    bar[pos] = "#"
    print("\r[" + "".join(bar) + "]", end="", flush=True)
    time.sleep(0.05)
    pos += direction
    if pos == 0 or pos == width - 1:
        direction *= -1
print()

print("\nDone!")

while True:
    reply = input("------Press Enter to continue-----").strip().lower()
    if reply in ("y"):
        break
    print("-----Please press enter to continue-----")
print()