import slackweb


def upload_message_to_slack(label):
    # 投稿者の名前設定を定義する
    YOUR_NAME = "Your name: XXXXX"

    # 検出したラベルを含めたメッセージを作成
    message = YOUR_NAME+": Detect ["+label+"]!!"

    # Slackにメッセージを送信する
    API_KEY = "https://hooks.slack.com/services/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"
    slack = slackweb.Slack(url=API_KEY)
    slack.notify(text=f"{YOUR_NAME}'s message: {message}")

    print("Send message to Slack: "+message)