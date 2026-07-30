"""
fabfile.py — automatiza el deploy y ejecucion del pipeline en el servidor.

Requisitos en tu maquina local:
    pip install fabric

Requisito previo (una sola vez): autenticacion SSH por llave configurada
hacia el servidor (ver instrucciones aparte: ssh-keygen + ssh-copy-id).

CONFIGURACION: ajusta las 3 variables de abajo (HOST, USER, PROJECT_DIR)
antes de usar.

USO (desde tu maquina local, parado en la carpeta del proyecto):
    fab deploy      # git pull + instala dependencias nuevas en el servidor
    fab extract     # corre extract_data.py en el servidor
    fab train       # lanza train_forecasting_tiered.py dentro de tmux (background)
    fab status      # muestra si el entrenamiento sigue corriendo y las ultimas lineas de log
    fab attach      # te conecta interactivamente a la sesion tmux (para ver en vivo)
    fab stop        # detiene la sesion de entrenamiento
"""
import os
from fabric import Connection, task

# =================================================================
# CONFIGURACION - AJUSTA ESTO
# =================================================================
HOST = "186.4.222.67"
PORT = 12980
USER = "udla"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_github_udla")   # ajusta si tu llave tiene otro nombre/ruta
PROJECT_DIR = "~/mia-inventory-forecasting-laferia"
VENV_ACTIVATE = f"source {PROJECT_DIR}/venv/bin/activate"
TMUX_SESSION = "forecasting"
LOG_FILE = f"{PROJECT_DIR}/training.log"
# =================================================================


def get_connection():
    return Connection(
        host=HOST,
        user=USER,
        port=PORT,
        connect_kwargs={"key_filename": SSH_KEY},
    )


@task
def deploy(c):
    """Trae los ultimos cambios de Git e instala dependencias nuevas."""
    conn = get_connection()
    with conn.cd(PROJECT_DIR):
        conn.run("git pull")
        conn.run(f"{VENV_ACTIVATE} && pip install -r requirements.txt")
    print("Deploy completado.")


@task
def extract(c):
    """Corre extract_data.py en el servidor (regenera parsed.json y abc_classification.json)."""
    conn = get_connection()
    with conn.cd(PROJECT_DIR):
        conn.run(f"{VENV_ACTIVATE} && python src/extraction/extract_data.py")


@task
def train(c):
    """
    Lanza train_forecasting_tiered.py dentro de una sesion tmux en background.
    Si la sesion ya existe (por ejemplo, de una corrida anterior interrumpida),
    no la duplica -- usa 'fab attach' para conectarte a la que ya esta corriendo.
    """
    conn = get_connection()
    existing = conn.run(f"tmux has-session -t {TMUX_SESSION}", warn=True, hide=True)
    if existing.ok:
        print(f"Ya existe una sesion tmux '{TMUX_SESSION}' corriendo. Usa 'fab attach' o 'fab status'.")
        return

    with conn.cd(PROJECT_DIR):
        cmd = (
            f"tmux new-session -d -s {TMUX_SESSION} "
            f"\"{VENV_ACTIVATE} && python -u src/forecasting/train_forecasting_tiered.py "
            f"2>&1 | tee {LOG_FILE}\""
        )
        conn.run(cmd)
    print(f"Entrenamiento lanzado en tmux (sesion '{TMUX_SESSION}'). Usa 'fab status' para ver el progreso.")

@task
def validate_b(c):
    conn = get_connection()
    existing = conn.run(f"tmux has-session -t validate_b", warn=True, hide=True)
    if existing.ok:
        print("Ya existe una sesion 'validate_b' corriendo.")
        return
    with conn.cd(PROJECT_DIR):
        cmd = (
            f"tmux new-session -d -s validate_b "
            f"\"{VENV_ACTIVATE} && python -u src/forecasting/validate_b_sample.py "
            f"2>&1 | tee validate_b.log\""
        )
        conn.run(cmd)
    print("Validacion lanzada en tmux (sesion 'validate_b').")

@task
def status(c):
    """Muestra si el entrenamiento sigue corriendo y las ultimas lineas relevantes del log."""
    conn = get_connection()
    result = conn.run(f"tmux has-session -t {TMUX_SESSION}", warn=True, hide=True)
    if result.ok:
        print(f"[EN CURSO] La sesion '{TMUX_SESSION}' sigue activa.\n")
    else:
        print(f"[NO ACTIVA] No hay una sesion '{TMUX_SESSION}' corriendo (termino o no se ha lanzado).\n")

    with conn.cd(PROJECT_DIR):
        conn.run(f"grep -v 'cmdstanpy' {LOG_FILE} | tail -n 20", warn=True)


@task
def attach(c):
    """
    Te conecta interactivamente a la sesion tmux para ver el progreso en vivo.
    Para salir SIN matar el proceso: Ctrl+B, luego D.
    """
    conn = get_connection()
    # Se abre una sesion interactiva real, no captura el output como los otros comandos
    conn.run(f"tmux attach -t {TMUX_SESSION}", pty=True)


@task
def stop(c):
    """Detiene la sesion de entrenamiento (mata el proceso)."""
    conn = get_connection()
    conn.run(f"tmux kill-session -t {TMUX_SESSION}", warn=True)
    print("Sesion detenida.")