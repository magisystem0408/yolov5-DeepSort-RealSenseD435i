import threading
import cv2
import time
import math
import glob
import numpy as np
import osascript
import pygame
from yeelight import Bulb
from yeelight import discover_bulbs

import handDetector as htm


class HandControl(object):
    def __init__(self, blob_ip: str = "192.168.1.4"):
        self.cap = cv2.VideoCapture(1)
        self.detector = htm.handDetector()
        self.main_mode = True
        self.light_mode = False
        self.music_mode = False
        self.music_play_mode = True
        self.volumeFlag = False
        self.islighting = False

        self.crashSize = 300
        self.musicList = glob.glob("sounds/musics/*.mp3")
        self.blob = Bulb(blob_ip)

        self._command_semaphore = threading.Semaphore(1)
        self._command_thread = None
        self.gesture()

    def _getLength(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def _calcAngle(self, right_x, right_y, middle_x, middle_y, top_x, top_y):
        radians = np.arctan2(top_y - middle_y, top_x - middle_x) - np.arctan2(right_y - middle_y, right_x - middle_x)
        angle = np.abs(radians * 180 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def send_command(self, command, blocking=True):
        self._command_thread = threading.Thread(
            target=self._send_command,
            args=(command, blocking)
        )
        self._command_thread.start()

    def _send_command(self, command, blocking=True):
        is_acquire = self._command_semaphore.acquire(blocking=blocking)
        if is_acquire:
            if command == "music_mode":
                pygame.mixer.music.load("sounds/announce/start_musicmode.wav")
                pygame.mixer.music.play(1)
                time.sleep(2)
                self._command_semaphore.release()

            if command == "light_mode":
                pygame.mixer.music.load("sounds/announce/start_lightmode.wav")
                pygame.mixer.music.play(1)
                time.sleep(2)
                self._command_semaphore.release()

            if command == "music_play":
                pygame.mixer.music.load(self.musicList[1])
                pygame.mixer.music.play(1)
                self._command_semaphore.release()

            if command == "music_stop":
                pygame.mixer.music.stop()
                self._command_semaphore.release()
            if command == "turn_on":
                self.blob.turn_on()

                self._command_semaphore.release()

            if command == "turn_off":
                self.blob.turn_off()
                self._command_semaphore.release()
        else:
            print("スキップします。")

    def gesture(self):
        pTime = 0
        while True:
            success, img = self.cap.read()
            height, width = img.shape[:2]
            img = self.detector.findHand(img)
            lmList = self.detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                # 親指
                right_thumb_tip_x, right_thumb_tip_y = lmList[4][1], lmList[4][2]
                # 親指のした
                right_thumb_ip_x, right_thum_ip_y = lmList[3][1], lmList[3][2]
                # 更にした
                right_thumb_cmc_x, right_thum_cmc_y = lmList[2][1], lmList[2][1]

                # 人差し指
                right_index_finger_tip_x, right_index_finger_tip_y = lmList[8][1], lmList[8][2]
                # 人差し指のした
                right_index_finger_dip_x, right_index_finger_dip_y = lmList[7][1], lmList[7][2]
                # 人差しの付け根より一つ上
                right_index_finger_pip_x, right_index_finger_pip_y = lmList[6][1], lmList[6][2]
                # 人差しの付け根より上
                right_index_finger_mcp_x, right_index_finger_mcp_y = lmList[5][1], lmList[5][2]

                # 中指
                right_middle_finger_tip_x, right_middle_finger_tip_y = lmList[12][1], lmList[12][2]
                # 中指のした
                right_middle_finger_dip_x, right_middle_finger_dip_y = lmList[11][1], lmList[11][2]
                # 中指の付け根より一つ上
                right_middle_finger_pip_x, right_middle_finger_pip_y = lmList[10][1], lmList[10][2]
                # 中指の付け根
                right_middle_finger_mcp_x, right_middle_finger_mcp_y = lmList[9][1], lmList[9][2]

                # 薬指
                right_ring_finger_tip_x, right_ring_finger_tip_y = lmList[16][1], lmList[16][2]
                # 薬指のした
                right_ring_finger_dip_x, right_ring_finger_dip_y = lmList[15][1], lmList[15][2]
                # 薬指の付け根のしたより一つ上
                right_ring_finger_pip_x, right_ring_finger_pip_y = lmList[14][1], lmList[14][2]
                # 薬指の付け根
                right_ring_finger_mcp_x, right_ring_finger_mcp_y = lmList[13][1], lmList[13][2]

                # 小指
                right_pinky_tip_x, right_pinky_tip_y = lmList[20][1], lmList[20][2]
                # 小指のした
                right_pinky_dip_x, right_pinky_dip_y = lmList[19][1], lmList[19][2]
                # 小指の付け根より一つ上
                right_pinky_pip_x, right_pinky_pip_y = lmList[18][1], lmList[18][2]
                # 小指の付け根
                right_pinky_mcp_x, right_pinky_mcp_y = lmList[17][1], lmList[17][2]

                # 長さ判定
                thumb_index_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                     right_index_finger_tip_x, right_index_finger_tip_y)
                thumb_middle_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                      right_middle_finger_tip_x, right_middle_finger_tip_y)
                thumb_ring_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                    right_ring_finger_tip_x, right_ring_finger_tip_y)

                thumb_pinky_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                     right_pinky_tip_x, right_pinky_tip_y, )

                # 角度計算(大回りよう)
                thumb_angle = self._calcAngle(right_thumb_cmc_x, right_thum_cmc_y,
                                              right_thumb_ip_x, right_thum_ip_y,
                                              right_thumb_tip_x, right_thumb_tip_y
                                              )

                index_angle = self._calcAngle(right_index_finger_mcp_x, right_index_finger_mcp_y,
                                              right_index_finger_pip_x, right_index_finger_pip_y,
                                              right_index_finger_tip_x, right_index_finger_tip_y
                                              )

                middle_angle = self._calcAngle(right_middle_finger_mcp_x, right_middle_finger_mcp_y,
                                               right_middle_finger_pip_x, right_middle_finger_pip_y,
                                               right_middle_finger_tip_x, right_middle_finger_tip_y
                                               )
                ring_angle = self._calcAngle(right_ring_finger_mcp_x, right_ring_finger_mcp_y,
                                             right_ring_finger_pip_x, right_ring_finger_pip_y,
                                             right_ring_finger_tip_x, right_ring_finger_tip_y,
                                             )

                pinky_angle = self._calcAngle(right_pinky_mcp_x, right_pinky_mcp_y,
                                              right_pinky_pip_x, right_pinky_pip_y,
                                              right_pinky_tip_x, right_pinky_tip_y
                                              )

                # 角度計算(小回り用)
                index_angle_2 = self._calcAngle(right_index_finger_pip_x, right_index_finger_pip_y,
                                                right_index_finger_dip_x, right_middle_finger_dip_y,
                                                right_index_finger_tip_x, right_index_finger_tip_y,
                                                )

                middle_angle_2 = self._calcAngle(right_middle_finger_pip_x, right_middle_finger_pip_y,
                                                 right_middle_finger_dip_x, right_middle_finger_dip_y,
                                                 right_middle_finger_tip_x, right_middle_finger_tip_y
                                                 )
                ring_angle_2 = self._calcAngle(right_ring_finger_pip_x, right_ring_finger_pip_y,
                                               right_ring_finger_dip_x, right_ring_finger_dip_y,
                                               right_ring_finger_tip_x, right_ring_finger_tip_y
                                               )
                pinky_angle_2 = self._calcAngle(right_pinky_pip_x, right_pinky_tip_y,
                                                right_pinky_dip_x, right_pinky_dip_y,
                                                right_pinky_tip_x, right_pinky_tip_y
                                                )

                firstGeture = index_angle >= 170 and thumb_ring_length <= 20 and thumb_middle_length <= 55 and thumb_pinky_length <= 50
                secondGesture = thumb_pinky_length <= 40 and thumb_ring_length <= 20 and index_angle >= 170 and middle_angle >= 170
                threeGesture = index_angle >= 170 and middle_angle >= 170 and ring_angle >= 170 and thumb_pinky_length <= 40
                yeyGeture = index_angle >= 170 and pinky_angle >= 170 and thumb_ring_length <= 40 and thumb_middle_length <= 40

                # チョキ(ミュージックモード)
                if secondGesture and self.main_mode == True:
                    self.send_command("music_mode")
                    self.main_mode = False
                    self.music_mode = True

                # ナンバーワン
                if firstGeture and self.main_mode == True:
                    self.send_command("light_mode")
                    self.main_mode = False
                    self.light_mode = True

                # モード選択
                if self.main_mode:
                    cv2.putText(img, "MainMode", (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

                # ライトモード
                if self.light_mode:
                    cv2.putText(img, "Light", (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(img, "NomalMode", (40, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                    if secondGesture and (not self.islighting):
                        self.islighting = True
                        self.send_command("turn_on")

                    if (self.islighting) and threeGesture:
                        self.islighting = False
                        self.send_command("turn_off")

                    # モードを抜ける
                    if yeyGeture:
                        self.light_mode = False
                        self.main_mode = True

                # ミュージックモード
                if self.music_mode:
                    cv2.putText(img, "Music", (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(img, "NomalMode", (40, 170), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                    # 再生する
                    if firstGeture and self.music_play_mode:
                        self.music_play_mode = False
                        self.send_command("music_play")
                        cv2.putText(img, "Play", (40, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                    # 停止する
                    if not self.music_play_mode:
                        print("値がfalseになりました。")
                    #     self.music_play_mode = True
                    #     self.send_command("music_stop")

                    # ボリューム調整
                    if threeGesture:
                        self.volumeFlag = True

                    if self.volumeFlag:
                        cv2.line(img, (right_thumb_tip_x, right_thumb_tip_y),
                                 (right_index_finger_tip_x, right_index_finger_tip_y),
                                 (255, 0, 255), 5)
                        cv2.rectangle(img, (width - self.crashSize, 0), (width, self.crashSize), color=(255, 255, 255),
                                      thickness=4)
                        vol = "set volume output volume " + str(np.interp(thumb_index_length, [20, 300], [0, 100]))
                        # osascript.osascript(vol)

                        # self.blob.set_brightness(np.interp(thumb_index_length, [20, 300], [0, 100]))



                        if right_index_finger_tip_x >= width - self.crashSize and right_index_finger_tip_y <= self.crashSize:
                            self.volumeFlag = False

                    # yey
                    if yeyGeture:
                        self.main_mode = True
                        self.music_mode = False

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS:{int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == '__main__':
    pygame.mixer.init(frequency=44100)
    pygame.mixer.music.load("sounds/announce/systemStart.wav")
    pygame.mixer.music.play(1)

    bulb_info = discover_bulbs()
    bulb_ip = bulb_info[0]["ip"]
    print("yeelightのIPアドレス" + bulb_ip)

    controller = HandControl(blob_ip=str(bulb_ip))
