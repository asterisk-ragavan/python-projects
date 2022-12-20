import random
import time

lst = range(100)
guess = [9, 18, 27, 36, 45, 54, 63, 72, 81]
symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
           "W", "X", "Y", "Z"]
choice = 'Y'

print("""Choose any two digit number, add together both the digits and then subtract the total from the original number. 
When you have the final number, look it up on the chart and find the letter next to it. While concentrating on 
the letter, press enter""")
while (choice == 'Y'):
    guess_letter = random.choice(symbols)
    for i in range(100):
        if i % 10 == 0:
            print("")
        if i in guess:
            print("{:5d} ".format(lst[i]), "-", guess_letter, " ", end="")
        else:
            print("{:5d} ".format(lst[i]), "-", random.choice(symbols), " ", end="")
    print("")
    temp = input("Press Enter to continue...")
    print("keep concentrating on the letter...")
    time.sleep(6)
    print("almost done")
    time.sleep(3)
    print("the letter next to ur guessed number is ", guess_letter)
    choice = input("do you want to try again [Y]es or [N]o").capitalize()
