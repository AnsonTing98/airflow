from joblib import load

model = load("animalModel.pkl")

print("***************Animal Predict App***************\n")
print("Instruction: Enter 1 for Yes, 0 for No\n")
print("************************************************\n")
while True:
    hair = int(input("Does the animal has hair? "))
    feather = int(input("Does the animal has feather? "))
    egg = int(input("Does the animal lay egg? "))
    milk = int(input("Does the animal has milk? "))
    airborne = int(input("Can the animal fly? "))
    aquatic = int(input("Is the animal aqutic (in water)? "))
    predator = int(input("Is the animal predator? "))
    toothed = int(input("Does the animal has toothed? "))
    backbone = int(input("Does the animal has backbone? "))
    breathes = int(input("Does the animal breathes? "))
    venomous = int(input("Does the animal venomous? "))
    fins = int(input("Does the animal has fins? "))
    legs = int(input("How many legs (0, 2, 4, 6, 8)? "))
    tail = int(input("Does the animal has tail? "))
    domestic = int(input("Does the animal domestic? "))
    catsize = int(input("Does the animal cat size?"))

    result = model.predict([[hair, feather, egg, milk, airborne, aquatic, predator, toothed, backbone, breathes,
           venomous, fins, legs, tail, domestic, catsize]])

    print("\nThe Animal Type is ", result)

    choice = input("Do you want to continue [y/N]: ")
    if(choice == "N" or choice == "n"):
        break
    elif(choice == "Y" or choice == "y"):
        print()
    else:
        break
