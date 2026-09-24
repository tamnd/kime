"""Records what Laya's `clean_email_body` answers, for kime-core's email test.

Run from a checkout of https://github.com/NandhaKishorM/laya with Laya's dependencies installed:

    python record.py /path/to/laya > crates/kime-core/tests/email/laya.json

It runs Laya's own `tests/test_email.py` with `clean_email_body` wrapped, so every body those
tests clean is kept with its answer, and then cleans a generated corpus of emails built from
the pieces the cleaner looks for: quote headers, sign-offs, device footers and disclaimers in
English, Portuguese and Spanish, with CRLF, quoted lines, odd whitespace and random cuts.
"""
import json
import os
import random
import runpy
import sys

root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)
import laya.email as E  # noqa: E402

seen = []
real = E.clean_email_body


def wrapped(body, max_chars=3000):
    out = real(body, max_chars)
    seen.append({"body": body, "max_chars": max_chars, "clean": out})
    return out


E.clean_email_body = wrapped
stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
try:
    runpy.run_path(os.path.join(root, "tests", "test_email.py"), run_name="__main__")
except SystemExit:
    pass
sys.stdout = stdout
fixtures = [dict(s, source="laya tests/test_email.py") for s in seen if isinstance(s["body"], str)]

# The committed file uses the defaults. A bigger run: record.py /path/to/laya 50000 7
size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 20260924)
REQUESTS = [
    "My account is locked, please unlock it.",
    "Could you refund invoice 4411?",
    "Preciso do boleto de setembro, o valor está errado.",
    "Necesito cancelar mi suscripción antes del viernes.",
    "The export to CSV fails with error 500 since Monday.",
    "Thanks for the quick reply. Could you also send the contract?",
    "Obrigado pelo retorno, mas o problema continua.",
    "De: 10/09 a 15/09 estarei de férias, preciso aprovar antes.",
    "Is this confidential? I need to know before I forward it.",
    "Confidential: I need a refund for order 5521.",
    "antes de imprimir o boleto, confira o valor",
    "¿Pueden revisar la factura 889? El importe no coincide.",
    "我的账户被锁定了，请帮忙解锁。",
    "Мой заказ 7781 не пришёл, верните деньги.",
    "Łukasz asked me to follow up on the renewal.",
]
QUOTES = [
    "On Mon, 3 Mar 2025 at 10:00, Bob Smith <bob@example.com> wrote:",
    "On Tuesday Bob wrote:",
    "Em seg., 3 de mar. de 2025 às 10:00, Maria <maria@exemplo.com.br> escreveu:",
    "Em resposta ao que você escreveu:",
    "El lun, 3 mar 2025 a las 10:00, Juan <juan@ejemplo.es> escribió:",
    "-----Original Message-----",
    "---------- Forwarded message ---------",
    "-----Mensagem original-----",
    "-----Mensaje reenviado-----",
    "________________________________",
    "From: Bob Smith <bob@example.com>",
    "De: Maria Souza <maria@exemplo.com.br>",
    "De: Maria Souza",
    "maria.souza@exemplo.com.br> escreveu:",
    "bob@example.com> wrote:",
]
AFTER_DE = ["Enviado: segunda-feira, 3 de março de 2025 10:00", "Enviado el: lunes", "Data: 03/03/2025 2025", "Para: 15/09"]
SIGNS = [
    "--", "Thanks,", "Thanks, Anna", "Best regards,", "Kind regards, Ana Lima", "Cheers", "Thank you so much!",
    "Many thanks", "Regards, Łukasz", "Thanks for the quick reply.", "Sincerely,", "Warmest regards", "Best and regards,",
    "Atenciosamente,", "Att,", "Abraços", "Obrigado desde já!", "Muito obrigada", "Cordialmente", "Saludos cordiales",
    "Muchas gracias de antemano", "Sent from my iPhone", "Enviado do meu iPhone", "Enviado do meu smartphone Samsung Galaxy.",
    "Get Outlook for iOS", "Obter o Outlook para Android", "Enviado desde mi móvil", "Enviado do meu celular o comprovante ontem.",
]
DISCLAIMERS = [
    "This email is confidential and intended solely for the named addressee.",
    "The information in this message is confidential and may be privileged.",
    "If you have received this message in error, please delete it.",
    "Esta mensagem é confidencial e destinada exclusivamente ao destinatário.",
    "Este mensaje es confidencial y privilegiado.",
    "Uso exclusivo del destinatario indicado.",
    "Se você recebeu esta mensagem por engano, apague-a.",
    "Si ha recibido este mensaje por error, bórrelo.",
    "Antes de imprimir, pense no meio ambiente.",
    "Pense no meio ambiente antes de imprimir.",
    "This communication is confidential\nand intended for the recipient only.",
]
NOISE = ["", " ", "\t", "  ", " ", "　", "\x1c", "\x1f", " ", "\x0b", "\x0c", "\x85"]


def line():
    k = rng.random()
    if k < 0.35:
        return rng.choice(REQUESTS)
    if k < 0.5:
        return rng.choice(SIGNS)
    if k < 0.6:
        return rng.choice(DISCLAIMERS)
    if k < 0.7:
        return "> " + rng.choice(REQUESTS)
    if k < 0.8:
        return rng.choice(QUOTES)
    if k < 0.85:
        return rng.choice(AFTER_DE)
    if k < 0.9:
        return ""
    return "".join(rng.choice("abcdeABCDE .,!?:@<>-_\n\r\t0123ñçãé漢\x1c") for _ in range(rng.randint(1, 40)))


def email():
    lines = [rng.choice(["Hi,", "Olá,", "Hola,", "Hello team,", ""])]
    for _ in range(rng.randint(1, 14)):
        lines.append(rng.choice(NOISE) + line() + rng.choice(NOISE))
    text = rng.choice(["\n", "\r\n", "\r", "\\n", "\n\n"]).join(lines)
    if rng.random() < 0.1:
        text = text * rng.randint(2, 4)
    if rng.random() < 0.2:
        c = rng.randint(0, len(text))
        text = text[:c]
    return text


corpus = []
for i in range(size):
    body = email()
    max_chars = rng.choice([3000, 3000, 3000, 200, 50, 1, 0])
    corpus.append({"body": body, "max_chars": max_chars, "clean": real(body, max_chars), "source": "generated"})

json.dump(fixtures + corpus, sys.stdout, ensure_ascii=False, indent=0)
sys.stdout.write("\n")
