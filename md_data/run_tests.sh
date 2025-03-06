#!/bin/bash
# Скрипт запуска тестов для модели MobileDepth

echo "Запуск тестов для модели MobileDepth..."
python3 -m unittest discover -s . -p "test_depth_model.py" -v

# Если вы хотите проверять код возврата и выводить сообщение
if [ $? -eq 0 ]; then
    echo "Все тесты прошли успешно!"
else
    echo "Некоторые тесты завершились с ошибкой!"
    exit 1
fi
