import random

default = ["bujju kutty ", "chella kutti", "kutty papaa", "sweety", "lucky charm"]
n = input("enter the names seperaated by , : ").split(sep=",")
if '0' in n:
    names = default
else:
    names = n
sorry = "sorry da chella kutti"
please = "plezeee da chella kutti"
emoji = ["❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "🤎", "💖", "💗", "💓", "💞", "💕", "❣️", "❤‍🩹", "❤‍🔥",
         "💝", "💘", "🌸", "🌺", "🌹", "💐", "🥰", "😘", "😍", "🫶🏻", "🥺"]
ip = input("love or sorry or please : ")
for i in range(int(input('How many times you wanna print: '))):
    n = random.randint(0, len(names) - 1)
    e = random.randint(0, len(emoji) - 1)
    if ip == "love":
        print(names[n] + " " + emoji[e] + " ", end="")
    if ip == "sorry":
        print("sorry da " + names[n] + " " + emoji[e] + " ", end="")
    if ip == "please":
        print("plezeee da " + names[n] + " " + emoji[e] + " ", end="")
