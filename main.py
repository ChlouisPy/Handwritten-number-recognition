"""
Handwritten number recognition program

With this program, you can create your own data, train your own model or use existing models.
You can save and load data and you can use your model to make predictions
GitHub : ChlouisPy
Twitter : @ChlouisPy
"""

import pygame
import numpy as np
import time
import datetime
import sys

# for the gui
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import messagebox
import matplotlib.pyplot as plt

# for the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, LeakyReLU, Flatten, Dropout
from tensorflow.keras.utils import to_categorical


# GPU
print(f"{datetime.datetime.now()}: Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

DEVICE: str = '/GPU:0'

# For the dimensions of numbers
DRAW_SIZE: int = 28

# initialisation of Pygame
pygame.init()

# factor by which will be multiplied the size of numbers for the size of the window
WINDOW_FACTOR = 20

# color for the window
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# lists that contains all drawings and outputs
X_TRAIN: list = []
Y_TRAIN: list = []

# model configuration
# epochs and verbose base
EPOCHS: int = 50
VERBOSE: int = 1


# create the model
class Model:
    """
    class which contains neural network for image recognition and also functions to import or export a neural network
    """

    def __init__(self):
        # main neural network
        self.model = Sequential([
            Conv2D(64, kernel_size=(3, 3), input_shape=(DRAW_SIZE, DRAW_SIZE, 1,)),
            LeakyReLU(),
            MaxPooling2D(),

            Conv2D(64, kernel_size=(3, 3)),
            LeakyReLU(),
            MaxPooling2D(),

            Flatten(),
            Dropout(0.5),

            Dense(64),
            LeakyReLU(),
            Dense(32),
            LeakyReLU(),
            Dense(10, activation="softmax")
        ])

        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]) # ppre train 2
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"]) # pre train 3
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"]) # pre train 1

    def save(self) -> None:
        """
        Function that will save the neural network as an .h5 file
        :return:
        """
        # get path
        location = askdirectory()
        print(f"{datetime.datetime.now()}: Saved in {location}")
        # creat path
        path = str(location) + "/" + str(int(time.time())) + ".h5"
        try:
            self.model.save(path)
            print(f"{datetime.datetime.now()}: Model successfully loaded")
        except ValueError as e:
            print(f"{datetime.datetime.now()}: Error during model saving {e}")

    def load(self) -> None:
        """
        Function that will load a neural network from an .h5 file
        :return:
        """
        # get path
        path = askopenfilename()
        # init var
        print(f"{datetime.datetime.now()}: Load from {path}")
        try:
            self.model = tf.keras.models.load_model(path)
            print(f"{datetime.datetime.now()}: Neural network loaded")
        except ValueError as e:
            print(f"{datetime.datetime.now()}: Error during model loading {e}")


# drawing

def drawing_window():
    """
    This function launches a Pygame drawing window and allows the user to draw a number
    :return: The user s drawing as a array
    """

    board = np.zeros((DRAW_SIZE, DRAW_SIZE))  # it is for de final picture

    screen_array = np.zeros((
        int(DRAW_SIZE * WINDOW_FACTOR + WINDOW_FACTOR),
        int(DRAW_SIZE * WINDOW_FACTOR + WINDOW_FACTOR)
    ))  # array for the drawing window

    # window for drawing
    screen = pygame.display.set_mode((DRAW_SIZE * WINDOW_FACTOR, DRAW_SIZE * WINDOW_FACTOR))
    screen.fill((250, 250, 250))
    pygame.display.set_caption("Drawing board")

    running: bool = True

    while running:
        for event in pygame.event.get():

            # clean the window if right click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                # fill the window of white
                screen.fill(WHITE)
                # reseth drawing array
                screen_array = np.zeros((
                    int(DRAW_SIZE * WINDOW_FACTOR + WINDOW_FACTOR),
                    int(DRAW_SIZE * WINDOW_FACTOR + WINDOW_FACTOR)
                ))  # array for the drawing window

            # Quit
            if event.type == pygame.QUIT:
                running = False

            # to draw
            if pygame.mouse.get_pressed() == (1, 0, 0):
                # si le boutton pour dessiner est pressé alors dessiner en noir sur la fenetre un cercle
                pygame.draw.circle(screen, BLACK, event.pos, WINDOW_FACTOR)

                # modifier sur l'array de dessin la valeur à cet eendroit
                # event.pos 0 = x 1 = y
                xc = event.pos[0]
                yc = event.pos[1]
                size = int(WINDOW_FACTOR)
                screen_array[absolute(yc - size):absolute(yc + size), absolute(xc - size):absolute(xc + size)] = 255

            # if key on the keyboard is pressed
            if event.type == pygame.KEYDOWN:

                # if return is pressed go to prediction menu
                if event.key == pygame.K_RETURN:
                    running = False
                    # change the resolution of the draw
                    board = change_resolution(screen_array)

                    return board

                # if i is pressed show the draw
                elif event.key == pygame.K_i:

                    # afficher les deux dessins
                    fig, axs = plt.subplots(2)
                    axs[0].imshow(screen_array, cmap="gray")
                    axs[1].imshow(change_resolution(screen_array), cmap="gray")
                    plt.show()

        pygame.display.flip()  # update

    sys.exit()


def start_draw() -> None:
    """
    this function will retrieve the user's design and add this design to the design list
    :return: None
    """
    global X_TRAIN, Y_TRAIN

    draw = drawing_window()

    question_window(draw)


# menus GUIs

def question_window(draw) -> None:
    """
    this function allows the user to give what he has drawn and to make modifications on the data and on the model
    :return: the draw made by the user -1 if for None
    """
    window = Tk()
    window.title("What did you draw")

    # add menu with export, import, clean, train, fit, exit
    menubar = Menu(window)
    filemenu = Menu(menubar, tearoff=0)
    # file menu
    filemenu.add_command(label="Information", command=lambda: info())
    filemenu.add_command(label="Import", command=lambda: open_drawings())
    filemenu.add_command(label="Export", command=lambda: export_drawings())
    filemenu.add_separator()
    filemenu.add_command(label="Delete last", command=lambda: delete_last())
    filemenu.add_command(label="Clear", command=lambda: clear())
    filemenu.add_command(label="Exit", command=lambda: window.destroy())

    # neural network menu
    nnmenu = Menu(menubar, tearoff=0)

    nnmenu.add_command(label="Save NN", command=lambda: model.save())
    nnmenu.add_command(label="Open NN", command=lambda: model.load())
    nnmenu.add_separator()
    nnmenu.add_command(label="Fit", command=lambda: fit())
    nnmenu.add_command(label="Evaluate", command=lambda: use_model(draw))
    nnmenu.add_command(label="Evaluate list", command=lambda: evaluate())

    menubar.add_cascade(label="File", menu=filemenu)
    menubar.add_cascade(label="AI", menu=nnmenu)
    window.config(menu=menubar)

    q = Label(window, text="What did you draw ?", font=50)
    q.grid(row=0, column=0, padx=25, pady=(25, 10))

    answer_frame = Frame(window, bg="red")
    answer_frame.grid(row=1, column=0, padx=25, pady=(10, 25))

    def place_answer_btn(n):
        """
        Place all 10 buttons to answer what did you draw
        :param n: value of the button
        :return: None
        """
        b = Button(answer_frame,
                   text=" {} ".format(str(n)),
                   font=50,
                   bg="white",
                   command=lambda: action_button(n))
        b.grid(row=0, column=n)

    def action_button(n):
        """
        Add in memory the drawing with his answer and chose the window
        :param n: value of the drawing
        :return:
        """
        if n is not None:
            print(f"{datetime.datetime.now()}: Draw added in memory with a value of {n}")

        window.destroy()
        new_data(draw, n)

    # place each of 10 buttons to answer the question what did you draw
    for i in range(10):
        place_answer_btn(i)

    # button if the draw is bad
    b1 = Button(answer_frame,
                text=" - ",
                font=50,
                bg="white",
                command=lambda: action_button(None))
    b1.grid(row=0, column=10)

    window.mainloop()

    # new draw
    start_draw()


def main_menu() -> None:
    """
    main window that appears at startup
    :return: None
    """

    window = Tk()
    window.title("Menu")

    # all functions for buttons in the window
    def scratch():
        window.destroy()
        start_draw()

    # menu bar of the window
    menubar = Menu(window)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open", command=open_drawings)
    filemenu.add_command(label="Scratch", command=scratch)
    filemenu.add_separator()
    filemenu.add_command(label="Open NN", command=lambda: model.load())
    filemenu.add_command(label="Import", command=lambda: open_drawings())
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=lambda: window.destroy())
    menubar.add_cascade(label="File", menu=filemenu)

    # title and subtitle

    lt = Label(window, text="Handwritten number recognition", font=("Arial black", 18),
               bg="white")
    lt.grid(row=0, column=0, padx=10, pady=(10, 5))

    subt = Label(window, text="v1.0", font=("Arial", 9),
                 bg="white")
    subt.grid(row=1, column=0, padx=10, pady=(0, 10))

    # start button
    start_btn = Button(window, text="Start", font=("Arial", 14), command=scratch)
    start_btn.grid(row=2, column=0, padx=10, pady=(0, 10))

    window.config(menu=menubar, bg="white")
    # start the menu window
    window.mainloop()


# function for importation or exportation

def export_drawings() -> None:
    """
    this function allows to export all drawings in memory in to a file
    :return: None
    """
    path_export = askdirectory()

    path = f"{path_export}/{int(time.time())}.npz"
    try:
        np.savez(path, x_train=X_TRAIN, y_train=np.array([np.argmax(l) for l in Y_TRAIN]))

        print(f"{datetime.datetime.now()}: Exporting data finished")
    except ValueError as e:
        print(f"{datetime.datetime.now()}: Error : {e}")


def open_drawings() -> None:
    """
    this function allows to import already saved drawings
    :return: None
    """
    global X_TRAIN, Y_TRAIN

    # get the path
    file_path = askopenfilename()

    try:
        data = np.load(file_path, allow_pickle=True)

    except ValueError as e:
        print(f"{datetime.datetime.now()}: Error : {e}")
        return None

    # open data
    _X_TRAIN = data["x_train"]
    _Y_TRAIN = data["y_train"]

    _X_TRAIN = [
        np.array(a).reshape((DRAW_SIZE, DRAW_SIZE,)) for a in _X_TRAIN
    ]
    _Y_TRAIN = to_categorical(_Y_TRAIN)
    _Y_TRAIN = _Y_TRAIN.tolist()
    # load new data
    X_TRAIN += _X_TRAIN
    Y_TRAIN += _Y_TRAIN

    print(f"{datetime.datetime.now()}: Importing data finished")


# modify data

def delete_last() -> None:
    """
    delete the last drawings in memory
    :return: None
    """
    if 0 < len(X_TRAIN) == len(Y_TRAIN) and len(Y_TRAIN) > 0:
        del X_TRAIN[-1]
        del Y_TRAIN[-1]

        print(f"{datetime.datetime.now()}: The last drawings is deleted")


def clear() -> None:
    """
    this function deletes all drawings in memory
    :return: None
    """
    global X_TRAIN, Y_TRAIN

    print(f"{datetime.datetime.now()}: All drawings in memory are deleted")

    X_TRAIN = []
    Y_TRAIN = []


def new_data(draw, n) -> None:
    """
    this function adds a new drawing made by the user
    :return:
    """
    global Y_TRAIN, X_TRAIN

    if n is not None:
        y_train = np.array([0 for _ in range(10)])
        y_train[n] = 1

        X_TRAIN.append(
            np.array(draw).reshape((DRAW_SIZE, DRAW_SIZE, 1))
        )
        Y_TRAIN.append(y_train)


# use model

def fit() -> None:
    """
    Train the model with all drawings in memory
    :return: None
    """

    t = time.time()
    x_train = np.array([
        np.array(a).reshape((DRAW_SIZE, DRAW_SIZE, 1,)) / 255 for a in X_TRAIN
    ])
    y_train = np.array([np.array(a) for a in Y_TRAIN])

    print(f"{datetime.datetime.now()}: Train with {len(X_TRAIN)} drawings")

    with tf.device(DEVICE):
        model.model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, verbose=VERBOSE)

    print(f"{datetime.datetime.now()}: Training finished in {round(time.time() - t)}s")


def use_model(draw) -> None:
    """
    Use the model with the image that has just been drawn
    :return: None
    """
    # get the result of the model
    result = model.model.predict(
        np.array([draw.reshape(DRAW_SIZE, DRAW_SIZE, 1)]) / 255
    )[0].tolist()

    hight_result = result.index(max(result))

    messagebox.showinfo("Prediction", f"Number : {hight_result}")

    # print the best result and all result
    print("========================= Result =========================")
    print("Neural Network detect a {}".format(hight_result))
    print("")
    print("Global result :")
    for i in range(10):
        print('\t {} : {}'.format(i, result[i]))

    print("==========================================================")
    print("")


def evaluate() -> None:
    """
    this function evaluate the efficiency of the model with a batch of data in memory
    :return: None
    """

    x_train = np.array([
        np.array(a).reshape((DRAW_SIZE, DRAW_SIZE, 1,)) / 255 for a in X_TRAIN
    ])
    y_train = np.array([np.array(a) for a in Y_TRAIN])

    t = time.time()
    with tf.device(DEVICE):
        accuracy = model.model.evaluate(x_train, y_train, verbose=VERBOSE)[1]

    print(f"{datetime.datetime.now()}: Evaluation finished in {round(time.time() - t)}s")

    print(f"{datetime.datetime.now()}: accuracy : ", accuracy)

    messagebox.showinfo("Evaluate", f"Model accuracy : {accuracy}")


# other

def info() -> None:
    print("========================= Information =========================")
    print("{} drawings in memory".format(len(X_TRAIN)))
    print("Epoch set at {}".format(EPOCHS))
    print("Drawings resolution : {} x {}".format(DRAW_SIZE, DRAW_SIZE))
    print("===============================================================")
    print(model.model.summary())
    print("===============================================================")


def absolute(x: int) -> int:
    """
    this function prevents values that are not within the range [0 ; DRAW_SIZE * WINDOW_FACTOR]
    :param x:
    :return: int x [0 ; 28 * WINDOW_FACTOR]
    """
    if 0 <= x <= DRAW_SIZE * WINDOW_FACTOR:
        return x

    elif x > DRAW_SIZE * WINDOW_FACTOR:
        return DRAW_SIZE * WINDOW_FACTOR

    elif x < 0:
        return 0


def change_resolution(array):
    """
    Change the resolution of the pygame drawing to the size of the drawing for the model
    :param array: Pygame drawing with a resolution of DRAW_SIZE * WINDOW_FACTOR
    :return: array of dimension (DRAW_SIZE, DRAW_SIZE)
    """
    return_array = np.zeros((DRAW_SIZE, DRAW_SIZE))

    # for each pixels in new image
    for x in range(DRAW_SIZE):
        for y in range(DRAW_SIZE):

            # the list that containe all value for 1 pixels
            temp_average_list = []

            # for each pixels in Pygame image
            for xp in range(WINDOW_FACTOR):
                for yp in range(WINDOW_FACTOR):
                    # average of all pixels
                    temp_average_list.append(array[int(x * WINDOW_FACTOR + xp)][y * WINDOW_FACTOR + yp])

            # make the average of all pixels il Pygame image for a area of (WINDOW_FACTOR, WINDOW_FACTOR)
            return_array[x][y] = float(sum(temp_average_list) / len(temp_average_list))

    return return_array


if __name__ == '__main__':
    # create
    model = Model()
    main_menu()
