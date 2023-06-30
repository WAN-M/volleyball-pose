import logging
import sys

from colorama import Fore, Style


# 获取对象
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(message)s",
            datefmt='%Y-%m-%d %H:%M')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


# 通过静态成员方法来调用
class Log:
    logger = get_logger()

    @staticmethod
    def debug(msg):
        Log.logger.debug(Fore.WHITE + "[DEBUG]: " + str(msg) + Style.RESET_ALL)

    @staticmethod
    def info(msg):
        Log.logger.info(Fore.GREEN + "[INFO]: " + str(msg) + Style.RESET_ALL)

    @staticmethod
    def warning(msg):
        Log.logger.warning("\033[38;5;214m" + "[WARNING]: " + str(msg) + "\033[m")

    @staticmethod
    def error(msg):
        Log.logger.error(Fore.RED + "[ERROR]: " + str(msg) + Style.RESET_ALL)

    @staticmethod
    def critical(msg):
        Log.logger.critical(Fore.RED + "[CRITICAL]: " + str(msg) + Style.RESET_ALL)
