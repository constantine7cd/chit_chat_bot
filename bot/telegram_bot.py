import os.path
import pickle
from abc import ABC
from functools import wraps

from telegram import ChatAction
from telegram.ext import (CommandHandler, Filters, MessageHandler,
                          PicklePersistence, Updater)

from bot.utils import (clean_text, generate_responses, load_pipeline,
                       load_weights, pick_best_response, setup_logger)

logger = setup_logger(__name__)


def start_command(update, context):
    """Start a new dialogue when user sends the command "/start"."""

    logger.debug(f"{update.effective_message.chat_id} - User: /start")
    context.chat_data['turns'] = []
    update.message.reply_text("Just start texting me. "
                              "Type \"/reset\". "
                              "Make sure to send no more than one message per turn.")


def reset_command(update, context):
    """Reset the dialogue when user sends the command "/reset"."""

    logger.debug(f"{update.effective_message.chat_id} - User: /reset")
    context.chat_data['turns'] = []
    update.message.reply_text("Beep beep!")


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(
                chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)

        return command_func

    return decorator


def error(_, context):
    logger.warning(context.error)


class TelegramBot(ABC):
    _updater = None

    def start(self):
        logger.info("Starting the telegram bot...")

        self._updater.start_polling()
        self._updater.idle()


class ChitChatBot(TelegramBot):
    def __init__(self, **kwargs):
        general_params = kwargs.get('general_params', {})
        device = general_params.get('device', -1)
        seed = general_params.get('seed', None)
        debug = general_params.get('debug', False)

        generation_pipeline_kwargs = kwargs.get(
            'generation_pipeline_kwargs', {})

        weights_path = generation_pipeline_kwargs.pop("weights", None)
        generation_pipeline_kwargs = {**{
            'model': 'microsoft/DialoGPT-medium'
        }, **generation_pipeline_kwargs}

        generator_kwargs = kwargs.get('generator_kwargs', {})
        generator_kwargs = {**{
            'max_length': 1000,
            'do_sample': True,
            'clean_up_tokenization_spaces': True
        }, **generator_kwargs}

        chatbot_params = kwargs.get('chatbot_params', {})
        if 'telegram_token' not in chatbot_params:
            raise ValueError("Please provide `telegram_token`")

        continue_after_restart = chatbot_params.get(
            'continue_after_restart', True)
        data_filename = chatbot_params.get('data_filename', 'data.pkl')

        self._generation_pipeline_kwargs = generation_pipeline_kwargs
        self._generator_kwargs = generator_kwargs
        self._chatbot_params = chatbot_params
        self._ranker_dict = {}
        self._device = device
        self._seed = seed
        self._debug = debug

        self._generation_pipeline = load_pipeline(
            'text-generation', device=device, **generation_pipeline_kwargs)

        load_weights(self._generation_pipeline.model, weights_path)

        logger.info("Initializing the telegram bot...")
        if continue_after_restart:
            persistence = PicklePersistence(data_filename)
            self._updater = Updater(
                chatbot_params['telegram_token'], use_context=True, persistence=persistence)
            if os.path.isfile(data_filename):
                with open(data_filename, 'rb') as handle:
                    chat_data = pickle.load(handle)['chat_data']
                for chat_id, chat_id_data in chat_data.items():
                    if len(chat_id_data['turns']) > 0:
                        self._updater.bot.send_message(
                            chat_id=chat_id, text="I'm back! Let's resume...")
                    else:
                        self._updater.bot.send_message(
                            chat_id=chat_id, text="I'm live!")
        else:
            self._updater = Updater(
                chatbot_params['telegram_token'], use_context=True)

        dp = self._updater.dispatcher
        dp.add_handler(CommandHandler('start', start_command))
        dp.add_handler(CommandHandler('reset', reset_command))
        dp.add_handler(MessageHandler(Filters.text, self.message))
        dp.add_error_handler(error)

    @send_action(ChatAction.TYPING)
    def message(self, update, context):
        """Receive message, generate response, and send it back to the user."""

        max_turns_history = self._chatbot_params.get('max_turns_history', 2)

        if 'turns' not in context.chat_data:
            context.chat_data['turns'] = []
        turns = context.chat_data['turns']

        user_message = update.message.text
        if max_turns_history == 0:
            context.chat_data['turns'] = []

        turn = {
            'user_messages': [],
            'bot_messages': []
        }
        turns.append(turn)
        turn['user_messages'].append(user_message)
        logger.debug(
            f"{update.effective_message.chat_id} - User: {user_message}")

        prompt = ""
        from_index = max(len(turns) - max_turns_history - 1,
                         0) if max_turns_history >= 0 else 0
        for turn in turns[from_index:]:
            for user_message in turn['user_messages']:
                prompt += clean_text(user_message) + \
                    self._generation_pipeline.tokenizer.eos_token
            for bot_message in turn['bot_messages']:
                prompt += clean_text(bot_message) + \
                    self._generation_pipeline.tokenizer.eos_token

        bot_messages = generate_responses(
            prompt,
            self._generation_pipeline,
            seed=self._seed,
            debug=self._debug,
            **self._generator_kwargs
        )
        if len(bot_messages) == 1:
            bot_message = bot_messages[0]
        else:
            bot_message = pick_best_response(
                prompt,
                bot_messages,
                self._ranker_dict,
                debug=self._debug
            )
        turn['bot_messages'].append(bot_message)
        logger.debug(
            f"{update.effective_message.chat_id} - Bot: {bot_message}")
        update.message.reply_text(bot_message)
