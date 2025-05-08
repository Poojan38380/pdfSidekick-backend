from colorama import init, Fore, Style
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='logs.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize colorama
init(autoreset=True)

def delete_logs():
    with open('logs.log', 'w') as file:
        file.truncate(0)    

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def print_header(message:str)->None:
        timestamp = get_timestamp()
        logger.info(f"[{timestamp}] {message}")
        print(f"\n{Fore.CYAN}{Style.BRIGHT}[{timestamp}] {message}{Style.RESET_ALL}\n")

def print_error(message:str)->None:
        timestamp = get_timestamp()
        logger.error(f"[{timestamp}] {message}")
        print(f"{Fore.RED}[{timestamp}] {message}{Style.RESET_ALL}")

def print_success(message:str)->None:
    timestamp = get_timestamp()
    logger.info(f"[{timestamp}] {message}")
    print(f"{Fore.GREEN}[{timestamp}] {message}{Style.RESET_ALL}")

def print_warning(message:str)->None:
    timestamp = get_timestamp()
    logger.warning(f"[{timestamp}] {message}")
    print(f"{Fore.YELLOW}[{timestamp}] {message}{Style.RESET_ALL}")

def print_title(message:str)->None:
    timestamp = get_timestamp()
    logger.warning(f"[{timestamp}] {message}")
    print(f"\n\n{Fore.MAGENTA}{Style.BRIGHT}[{timestamp}] {message}{Style.RESET_ALL}\n")

def print_info(message:str)->None:
    timestamp = get_timestamp()
    logger.info(f"[{timestamp}] {message}")
    print(f"{Fore.BLUE}[{timestamp}] {message}{Style.RESET_ALL}")

def get_user_input(prompt:str)->str:
    return input(f"{Fore.YELLOW}{prompt}{Style.RESET_ALL}").strip()
