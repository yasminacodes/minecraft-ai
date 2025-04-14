import cv2 as cv
import numpy as np
import tensorflow as tf
import pyautogui
import random
import pytesseract
from imutils.object_detection import non_max_suppression
from PIL import Image
from pynput.keyboard import Controller, Key
import json
from datetime import datetime
import os

class Player:
    def __init__(self, game):
        if(game == "minecraft"):
            self.actionsKeys = [
                'space', 'shift', 'ctrl',
                'a', 'd', 's', 'w',
                'shift+w', 'ctrl+w', 'shift+q',
                '_left', '_middle', '_right', '_move',
                'move_left', 'move_right', 'move_up', 'move_down', 'move_center',
                'ctrl+_right', 'shift+_right',
                '1', '2', '3', '4',
                '5', '6', '7', '8', '9',
                'q', 'e', 'f'
            ]
            self.actions = len(self.actionsKeys)
        else:
            self.actions = None

        self.keyPressed = None
        self.keyboard_controller = Controller()
        
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

    def __centerMouse(self):
        centerX = (self.screenX0 + self.screenX1) // 2
        centerY = (self.screenY0 + self.screenY1) // 2
        pyautogui.moveTo(centerX, centerY)

    def __play(self, action, image=None):
        actionKey = self.actionsKeys[action]
        self.lastAction = action

        keys = actionKey.split('+')

        if self.keyPressed and self.keyPressed not in keys:
            self.keyboard_controller.release(self.__getKey(self.keyPressed))
            self.keyPressed = None

        if any(k.startswith('_') for k in keys) or any(k.startswith('move_') for k in keys):
            if 'move_left' in keys or 'move_right' in keys or 'move_up' in keys or 'move_down' in keys or 'move_center' in keys:
                self.__centerMouse()

                dx, dy = 0, 0
                step = 30

                if 'move_left' in keys:
                    dx = -step
                elif 'move_right' in keys:
                    dx = step
                elif 'move_up' in keys:
                    dy = -step
                elif 'move_down' in keys:
                    dy = step

                print(f"Moving mouse relative: dx={dx}, dy={dy} ({actionKey})")
                pyautogui.moveRel(dx, dy, duration=0.05)

            elif '_move' in keys:
                targetX = random.randint(self.screenX0, self.screenX1)
                targetY = random.randint(self.screenY0, self.screenY1)
                print(f"Moving mouse to position: {targetX}, {targetY}")
                pyautogui.moveTo(targetX, targetY, duration=0.5)
            else:
                for key in keys:
                    if not key.startswith('_'):
                        self.keyboard_controller.press(self.__getKey(key))

                for key in keys:
                    if key == '_right' and image is not None:
                        self.preInteractionImage = image.copy()
                        pyautogui.click(button='right')
                    elif key == '_left':
                        pyautogui.click(button='left')
                    elif key == '_middle':
                        pyautogui.click(button='middle')

                for key in keys:
                    if not key.startswith('_'):
                        self.keyboard_controller.release(self.__getKey(key))

        else:
            print(f"Pressing keys: {keys}")
            for key in keys:
                self.keyboard_controller.press(self.__getKey(key))
            for key in reversed(keys):
                self.keyboard_controller.release(self.__getKey(key))

            if len(keys) == 1:
                self.keyPressed = keys[0]
            else:
                self.keyPressed = None



    def __getKey(self, key):
        key_map = {
            'enter': Key.enter,
            'tab': Key.tab,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'space': Key.space
        }
        return key_map.get(key, key)

    def __computeDiscountedRewards(self, rewards, gamma=0.99):
        discountedRewards = np.zeros_like(rewards)
        runningTotal = 0
        for t in reversed(range(len(rewards))):
            runningTotal = runningTotal * gamma + rewards[t]
            discountedRewards[t] = runningTotal
        return discountedRewards

    """
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

            if text in ['respawn', 'died', 'slain', 'score']:
                self.keyboard_controller.release(Key.tab)
                self.keyboard_controller.press(Key.tab)
                self.keyboard_controller.release(Key.enter)
                self.keyboard_controller.press(Key.enter)
                return -1
            elif text in ['menu', 'back', 'game']:
                self.keyboard_controller.release(Key.tab)
                self.keyboard_controller.press(Key.tab)
                self.keyboard_controller.release(Key.enter)
                self.keyboard_controller.press(Key.enter)
                return 0
        return 1
    """

    def __calculateReward(self, image):
        visual_reward = 0.0
        interaction_bonus = 0.0

        if hasattr(self, "lastImage") and self.lastImage is not None:
            diff = cv.absdiff(self.lastImage, image)
            mean_diff = np.mean(diff) / 255
            visual_reward = min(mean_diff * 10, 1.0)
        self.lastImage = image.copy()

        if hasattr(self, "lastAction") and self.lastAction is not None:
            if self.actionsKeys[self.lastAction] == "_right" and hasattr(self, "preInteractionImage"):
                interaction_diff = np.mean(cv.absdiff(self.preInteractionImage, image)) / 255
                if interaction_diff > 0.03:
                    interaction_bonus = 0.5
                    print("[Interaction] Successful right-click detected")

        newImg = cv.resize(image, (1280, 640))
        height, width, _ = newImg.shape
        rW = width / float(1280)
        rH = height / float(640)

        blob = cv.dnn.blobFromImage(image, 1.0, (width, height),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.readNet.setInput(blob)
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        (scores, geometry) = self.readNet.forward(layerNames)

        minConfidence = 0.9
        (numRows, numCols) = scores.shape[2:4]
        rects, confidences = [], []

        for y in range(numRows):
            scoresData = scores[0, 0, y]
            xData0, xData1, xData2, xData3, anglesData = (
                geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y],
                geometry[0, 3, y], geometry[0, 4, y]
            )
            for x in range(numCols):
                if scoresData[x] < minConfidence:
                    continue
                offsetX, offsetY = x * 4.0, y * 4.0
                angle = anglesData[x]
                cos, sin = np.cos(angle), np.sin(angle)
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
            endX = min(int(endX * rW), newImg.shape[1])
            endY = min(int(endY * rH), newImg.shape[0])
            region = newImg[startY:endY, startX:endX]
            if region.shape[0] > 0 and region.shape[1] > 0:
                text = pytesseract.image_to_string(region, config='--psm 6').lower().strip()

            if any(t in text for t in ['respawn', 'died', 'slain', 'score']):
                print("[State] Death screen detected")
                self.keyboard_controller.release(Key.tab)
                self.keyboard_controller.press(Key.tab)
                self.keyboard_controller.release(Key.enter)
                self.keyboard_controller.press(Key.enter)
                return -1.0 + visual_reward + interaction_bonus

            elif any(t in text for t in ['menu', 'back', 'game']):
                print("[State] Menu screen detected")
                self.keyboard_controller.release(Key.tab)
                self.keyboard_controller.press(Key.tab)
                self.keyboard_controller.release(Key.enter)
                self.keyboard_controller.press(Key.enter)
                return 0.0 + visual_reward + interaction_bonus

        print(f"[Reward] Visual: {visual_reward:.3f}, Interaction: {interaction_bonus:.3f}")
        return visual_reward + interaction_bonus


    def __logTrainingStats(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        stats = {
            "datetime": now,
            "episodes_done": self.episodesDone,
            "steps_per_episode": self.steps,
            "episode_rewards": self.episodeRewards,
            "average_reward": float(np.mean(self.episodeRewards)) if self.episodeRewards else 0.0
        }

        log_filename = f"{log_dir}/training_log_{now}.json"
        with open(log_filename, "w") as log_file:
            json.dump(stats, log_file, indent=4)

        print(f"Estadísticas guardadas en: {log_filename}")

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

        model_path = 'models/playerai_model.keras'
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        if os.path.exists(model_path):
            print("Existing model found, loading for accumulative training...")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("No existing model found, creating a new one...")
            self.model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255, input_shape=self.shape),
                tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(self.actions, activation='linear')  # acción más prometedora
            ])


    def trainModel(self, image):
        if self.stepsDone > 0 or self.episodesDone > 0:
            reward = self.__calculateReward(image)
            print(f"Reward obtained: {reward}")
            self.episodeRewards.append(reward)

        if self.episodesDone >= self.episodes:
            os.makedirs("models", exist_ok=True)
            self.model.save('models/playerai_model.keras')
            self.__logTrainingStats()
            return False
        
        if self.stepsDone >= self.steps:
            discountedRewards = self.__computeDiscountedRewards(self.episodeRewards)
            print(f"Episode rewards: {discountedRewards}")

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
                normalizedProbs = tf.ones(self.actions) / self.actions
            else:
                normalizedProbs = probs / sumProbs

            normalizedProbs = normalizedProbs.numpy()

            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                action = np.random.choice(self.actions, p=normalizedProbs)
            
            self.__play(action, image)

            self.episodeActions.append(action)
            self.episodeImages.append(image)

            self.stepsDone += 1
        
        return True
