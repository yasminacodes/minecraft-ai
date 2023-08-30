from modules.Capturer import Capturer
from modules.Player import Player
from time import sleep

def main():
    try:
        print("The program started!")

        seg = Capturer()
        player = Player(game = "minecraft")

        screen, shape = seg.selectWindow()

        print("Configuring model...")
        player.configModelToTrain(screen, shape, 5, 5)
        print("Model configured: starting training")

        training = True
        while training:
            frame = seg.captureScreen()
            training = player.trainModel(frame)
            sleep(0.1)

        print("The program finished")
    except KeyboardInterrupt:
        print("The program finished")
    
    

if __name__ == "__main__":
    main()