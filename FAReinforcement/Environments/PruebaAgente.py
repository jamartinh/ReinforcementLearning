from .Agente import *
from PIL import Image
from PIL import ImageGrab

scene = display(title='Pursuit and Evasion  -Differential Games Simulator-',
                width=1025, height=770,
                center=(0,0,0), background=(1,1,1),
                autoscale=0,fullscreen=1)
contador=1
while True:

    #Pata
    Pata=Agente(radians(75),10,(randint(-5,5),randint(-5,5),0),color.orange,(-1,0,0),L=1.5)
    Pata.setac(radians(0),0.1)
    Pata.update()



    #Patito
    Patito=Agente(radians(80),10,(randint(-5,5),randint(-5,5),0),color.yellow,(0,1,0),L=1.5)
    Patito.setac(radians(0),0.05)
    Patito.update()

    #Zorro
    Zorro=Agente(radians(75),10,(randint(-5,5),randint(-5,5),0),color.red,L=2.0)
    Zorro.setac(radians(0),0.075)
    Zorro.update()


    while True:
        Pata.react(Zorro,1)


        Zorro.react(Patito,1)
        #Zorro.react(Pata,-1)

        Patito.react(Pata,1)
        #Patito.react(Zorro,-1)

        Pata.update()
        Zorro.update()
        Patito.update()
        if sqrt(sum(Patito.state.x-Zorro.state.x)**2)<0.5:
            #im = ImageGrab.grab()
            #im.save("example"+str(contador)+".jpg")
            del Pata
            del Patito
            del Zorro
            for o in scene.objects:
                o.visible=0

            break

        rate(30)
    contador=contador+1

