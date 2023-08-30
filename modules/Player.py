import cv2 as cv
import numpy as np
import tensorflow as tf
import keyboard
import pyautogui
import random
import pytesseract
from imutils.object_detection import non_max_suppression
from PIL import Image

class Player:
    def __init__(self, game):
        if(game == "minecraft"):
            self.actionsKeys = [
                'space', 'shift', 'ctrl',
                'a', 'd', 's', 'w',
                '_left', '_middle', '_right', '_move',
                '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                'q', 'e','f'

            ]
            self.actions = len(self.actionsKeys)
        else:
            self.actions = None
        
        self.keyPressed = None
        
        self.shape = None
        self.episodes = None
        self.steps = None
        self.epsilon = None

        self.model = None
        self.optimizer = None

        self.episodesDone = None
        self.stepsDone = None
        self.episodeRewards = None
        self.episodeActions = None
        self.episodeImages = None

        self.screenX0 = None
        self.screenY0 = None
        self.screenX1 = None
        self.screenY1 = None

        self.readNet = None
    
    def __computeLoss(self, probs, rewards):
        actionLogProbs = tf.math.log(probs)
        rewards = tf.cast(rewards, dtype=tf.float32)
        
        if len(rewards.shape) == 1:
            rewards = tf.expand_dims(rewards, axis=1)
        
        weightedLogProbs = actionLogProbs * rewards

        return -tf.reduce_mean(weightedLogProbs)

    def __play(self, action):
        actionKey = self.actionsKeys[action]

        if actionKey[0] == "_":
            if(actionKey == "_move"):
                targetX = random.randint(self.screenX0, self.screenX1)
                targetY = random.randint(self.screenY0, self.screenY1)
                print(f"Moving mouse to position: {targetX}, {targetY}")
                pyautogui.moveTo(targetX, targetY, duration=0.5)
            else:
                actionClick = actionKey.replace("_", "")
                print(f"Clicking {actionClick}")
                pyautogui.click(button=actionClick)
        else:
            print(f"Pressing {actionKey}")
            if self.keyPressed != actionKey:
                keyboard.release(actionKey)
                keyboard.press(actionKey)
                self.keyPressed = actionKey
            
    
    def __computeDiscountedRewards(self, rewards, gamma=0.99):
        discountedRewards = np.zeros_like(rewards)
        runningTotal = 0
        for t in reversed(range(len(rewards))):
            runningTotal = runningTotal * gamma + rewards[t]
            discountedRewards[t] = runningTotal
        return discountedRewards
    
    def __calculateReward(self, image):
        originalHeight, originalWidth, _ = image.shape
        
        newImg = cv.resize(image, (1280, 640))
        height, width, _ = newImg.shape

        rW = width / float(1280)
        rH = height / float(640)

        blob = cv.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        self.readNet.setInput(blob)
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        (scores, geometry) = self.readNet.forward(layerNames)

        minConfidence = 0.9
        dead = False
        paused = False

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < minConfidence:
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            region = newImg[startY:endY, startX:endX]
            text = pytesseract.image_to_string(region, config='--psm 6')
            text = text.lower().strip()
            #print(f"Text found: {text}")

            if text == 'respawn' or text == 'died' or text == 'slain' or text == 'score':
                keyboard.release('tab')
                keyboard.press('tab')
                keyboard.release('enter')
                keyboard.press('enter')
                return -1
            elif text == 'menu' or text == 'back' or text == 'game':
                keyboard.release('tab')
                keyboard.press('tab')
                keyboard.release('enter')
                keyboard.press('enter')
                return 0
        return 1


    ## PUBLIC ##
    def configModelToTrain(self, screen, shape, episodes, steps, epsilon = 0.5):
        self.shape = shape
        self.episodes = episodes
        self.steps = steps
        self.epsilon = epsilon

        self.screenX0 = screen[0]
        self.screenY0 = screen[1]
        self.screenX1 = screen[2]
        self.screenY1 = screen[3]

        self.episodesDone = 0
        self.stepsDone = 0
        self.episodeRewards = []
        self.episodeActions = []
        self.episodeImages = []

        self.readNet = cv.dnn.readNet('frozen_east_text_detection.pb')

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.actions, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    def trainModel(self, image):
        if self.stepsDone > 0 or self.episodesDone > 0:
            reward = self.__calculateReward(image)
            print(f"Reward obtained: {reward}")
            self.episodeRewards.append(reward)

        if self.episodesDone >= self.episodes:
            self.model.save('models/playerai_model')
            return False
        
        if self.stepsDone >= self.steps:
            discountedRewards = self.__computeDiscountedRewards(self.episodeRewards)
            print(f"Episode rewards: {discountedRewards}")

            #actionsOneHot = tf.one_hot(self.episodeActions, self.actions)
            minLen = min(len(self.episodeImages), len(discountedRewards))
            self.episodeImages = self.episodeImages[:minLen]
            discountedRewards = discountedRewards[:minLen]

            self.episodeImages = np.array(self.episodeImages, dtype=np.float32)
            
            with tf.GradientTape() as tape:
                logits = self.model(self.episodeImages, training=True)
                probs = tf.nn.softmax(logits, axis=1)
                
                if len(probs.shape) == 1:
                    probs = tf.expand_dims(probs, axis=0)

                loss = self.__computeLoss(probs, discountedRewards)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.episodesDone += 1
            self.stepsDone = 0
            self.episodeRewards = []
            self.episodeActions = []
            self.episodeImages = []
        else:
            probs = self.model(tf.convert_to_tensor(image[None, :], dtype=tf.float32))
            probs = tf.nn.softmax(probs[0])
            sumProbs = tf.reduce_sum(probs)

            if tf.math.is_nan(sumProbs) or sumProbs == 0:
                normalizedProbs = tf.ones(self.actions) / self.actions  # Distribución uniforme
            else:
                normalizedProbs = probs / sumProbs

            normalizedProbs = normalizedProbs.numpy()

            action = None
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                action = np.random.choice(self.actions, p=normalizedProbs)
            
            
            self.__play(action)
            
            self.episodeActions.append(action)
            self.episodeImages.append(image)

            self.stepsDone = self.stepsDone + 1
        
        return True

        

