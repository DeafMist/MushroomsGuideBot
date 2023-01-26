import io

import numpy
import telebot
from PIL import Image

import mushrooms_utils
from neural_network import mushrooms_model

with open('token/token.txt', 'r') as file:
    TOKEN = file.read()

bot = telebot.TeleBot(TOKEN)

model = mushrooms_model.model


@bot.message_handler(commands=['start'])
def start(message):
    text = f'Привет, {message.from_user.first_name} {message.from_user.last_name}!\n' \
           f'Меня зовут Шляпник.\nЯ бот, который был создан для помощи грибникам, ' \
           f'которые еще не преисполнились в познании и не могут отличить съедобный гриб от несъедобного.\n' \
           f'Все, что от тебя требуется - отправить мне фотографию гриба, а я дам тебе краткую сводку о том, ' \
           f'с чем ты имеешь дело.\n' \
           f'Положись на меня, и все будет хорошо - твой живот и твое здоровье будут спасены!!!'
    bot.send_message(message.chat.id, text)


@bot.message_handler(content_types=['text'])
def get_user_text(message):
    user_text = message.text.lower()
    if user_text == "привет" or user_text == 'ку' or user_text == 'здравствуй':
        text = 'И тебе привет!'
    elif user_text == 'пока':
        text = 'Всего хорошего!\n' \
                   'Буду ждать твоего возвращения!'
    else:
        text = 'Моя задача - определять тип гриба по фотографии.\n' \
               'Твой текст читать и обрабатывать нет никакого желания!\n' \
               'Кидай побыстрее фотографию, я жду!'
    bot.send_message(message.chat.id, text)


@bot.message_handler(content_types=['photo'])
def get_user_photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    image = numpy.array(Image.open(io.BytesIO(downloaded_file)).resize((128, 128))) / 255
    prediction, probability = mushrooms_utils.predict(image, model)
    text = f'Это - {prediction} с {numpy.around(probability, 2)}% вероятностью.\n' \
           f'Краткая сводка из Википедии:\n' \
           f'{mushrooms_utils.DESCRIPTIONS[prediction]}'
    bot.send_message(message.chat.id, text)


bot.polling(none_stop=True)
