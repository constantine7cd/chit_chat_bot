### Instructions

#### How to run bot (Unix, MacOS)?
 1. Create your own environment with conda/pip:
    
    ```bash
    # Anaconda
    $ conda create -n chit_chat_bot python=3.8.2
    $ conda activate chit_chat_bot

    # Using venv 
    $ python -m venv chit_chat_bot
    $ source chit_chat_bot/bin/activate
    ```

 2. Install packages:
    ```bash
    $ pip install -r requirements.txt
    ```

    If you failed with the last command then try to install packages manually. 
    Mac M1 users, please, follow this installation [instruction](https://towardsdatascience.com/yes-you-can-run-pytorch-natively-on-m1-macbooks-and-heres-how-35d2eaa07a83);

3. Register a new telegram bot: [link](https://core.telegram.org/bots#6-botfather);
4. After successfull bot installation copy your bot **token** and paste to ```configs/gpt-medium.yaml```:
    ```yaml
    chatbot_params:
        max_turns_history: 2
        telegram_token: <paste here token>
    ```

5. Download model weights from the following [link](https://drive.google.com/drive/folders/1MaqehaDoA8D5V8PROil3Q_eXObCDikfL?usp=sharing);
6. Copy weights to folder ```weights```;
7. Finally, run bot:
    ```
    ./run_bot_gpt_medium.sh 
    ```




