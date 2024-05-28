
import requests
import sys


def sendMessageToHeathcliff05botToAChat(chatId,message):
       heathcliff05_bot_url = "https://api.telegram.org/<config>"
       requests.get(heathcliff05_bot_url+"sendMessage?chat_id={chatId}&text={message}".format(chatId=chatId,message=message))

# if __name__== "__main__":
#     chatid = ""
#     message = ""
#     if len(sys.argv) > 1:
#         l = len(sys.argv)
#         for i in range(l):
#             if sys.argv[i] == "--chatid":
#                 arg = sys.argv[i+1]
#                 chatid = arg
#             if sys.argv[i] == "--message":
#                 arg = sys.argv[i+1]
#                 message = arg
#     sendMessageToHeathcliff05_botToAChat(chatid,message)
