import smtplib
from email.message import EmailMessage
from app.core.config import settings

def send_email(to_email: str, subject: str, content: str):
    """
    Envoie un email simple en texte brut.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.FROM_EMAIL
    msg["To"] = to_email
    msg.set_content(content)

    with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        smtp.send_message(msg)
