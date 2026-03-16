Proof from redis:
```
127.0.0.1:6379> JSON.GET "chat:session-4fc53bc7:01KKW6HG71YN78KESK81ZBK14B"
"{\"type\":\"tool\",\"message_id\":\"01KKW6HG71YN78KESK81ZBK14B\",\"data\":{\"content\":\"Model: Samsung Galaxy S24 Ultra\\nPrice: $1379.2\\nRating: 85.0\\nSIM: Dual Sim, 3G, 4G, 5G, VoLTE, Vo5G, Wi-Fi\\nProcessor: Snapdragon 8 Gen 3, Octa Core Processor\\nRAM: 12 GB RAM, 256 GB inbuilt\\nBattery: 5100 mAh Battery with 45W Fast Charging\\nDisplay: 6.83 inches, 1200 x 2860 px, 144 Hz Display with Punch Hole\\nCamera: 200 MP Quad Rear & 60 MP Front Camera\\nCard: Memory Card Not Supported\\nOS: Android v13\\nIn Stock: True\",\"additional_kwargs\":{},\"type\":\"tool\",\"tool_call_id\":\"call_1wUMIczOC9UBlS3v2piwqnpN\",\"status\":\"success\"},\"session_id\":\"session-4fc53bc7\",\"timestamp\":1773694140.641643}"
127.0.0.1:6379> JSON.GET "chat:session-d39a822e:01KKW6XN7JCSASQSRN4B0S8ZKP"
"{\"type\":\"human\",\"message_id\":\"01KKW6XN7JCSASQSRN4B0S8ZKP\",\"data\":{\"content\":\"S24 Ultra and Ihpone 13\",\"additional_kwargs\":{},\"type\":\"human\"},\"session_id\":\"session-d39a822e\",\"timestamp\":1773694538.99408}"
```
