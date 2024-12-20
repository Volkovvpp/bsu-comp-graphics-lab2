# Обработка изображений с помощью OpenCV

Приложение на Python для обработки изображений с использованием библиотеки OpenCV и графического интерфейса tkinter. Оно позволяет выполнять базовые операции по обработке изображений, включая поэлементные операции, линейное контрастирование, метод пороговой обработки Оцу, градиентную пороговую обработку, а также сброс изображения к исходному состоянию.

## Описание

### Возможности:
- Загрузка изображения с устройства.
- Применение линейного контрастирования.
- Применение пороговой обработки методом Оцу.
- Применение градиентной пороговой обработки.
- Сброс изображения к оригиналу.

## Установка

1. Убедитесь, что у вас установлен Python версии 3.6 или выше.
2. Установите необходимые зависимости с помощью команды:

    ```bash
    pip install opencv-python-headless pillow numpy
    ```

## Запуск приложения

Запустите приложение с помощью команды:

    ```bash
    python app.py
    ```

После запуска откроется графический интерфейс, в котором можно загружать и обрабатывать изображения.

## Использование

### Основной интерфейс

- **Выбрать изображение**: позволяет загрузить изображение из файловой системы.
- **Поэлементные операции и линейное контрастирование**: выполняет линейное контрастирование изображения.
- **Метод Оцу**: применяет метод пороговой обработки Оцу.
- **Градиентная пороговая обработка**: применяет градиентную пороговую обработку.
- **Сбросить изображение**: восстанавливает изображение до исходного состояния.

## Примечание

Данное приложение использует библиотеку OpenCV для выполнения обработки изображений, а также PIL для отображения изображений в интерфейсе tkinter. Перед началом работы убедитесь, что выбранное изображение имеет допустимый формат (например, JPG, PNG).

## Примеры

### Линейное контрастирование

- Применяет изменение яркости и контрастности по каналам изображения.

### Метод Оцу

- Автоматический порог для бинаризации изображения, основанный на гистограмме.

### Градиентная пороговая обработка

- Использует метод расчета градиента для выделения границ на изображении.
