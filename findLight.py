from yeelight import Bulb, discover_bulbs

blub_info = discover_bulbs()
blob =Bulb("192.168.1.3")
blob.set_rgb(0,0,255)
print(blub_info)
