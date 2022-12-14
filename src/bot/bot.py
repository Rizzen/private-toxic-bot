import logging
import random
import model
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes

model_class = model.Model("koolKat/iro_model")
reply_possibility = 0.1
bot_id = 5777831507


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rd = random.random()
    text = update.message.text
    if text is None:
        return
    text = text.lower()
    model_class.process_message(text)

    if 'славик' in text:
        await generate_and_send2(update, context)
    elif "@glory_iro_bot" in text:
        await generate_and_send2(update, context)
    elif update.message.reply_to_message is not None:
        if update.message.reply_to_message.from_user.id == bot_id:
            await generate_and_send2(update, context)
    elif rd <= reply_possibility:
        await generate_and_send2(update, context)

async def generate_and_send2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    answer = model_class.predict2(text)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=answer,
        reply_to_message_id=update.message.id)


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    application = ApplicationBuilder().token('5777831507:AAFGj7SO53LKp6961939TVrJsFXxRPDV9Og').build()
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(echo_handler)
    application.run_polling()