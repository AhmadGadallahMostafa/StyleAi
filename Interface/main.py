################################################################################
##
## BY: WANDERSON M.PIMENTA
## PROJECT MADE WITH: Qt Designer and PySide2
## V: 1.0.0
##
## This project can be used freely for all uses, as long as they maintain the
## respective credits only in the Python scripts, any information in the visual
## interface (GUI) can be modified without any implication.
##
## There are limitations on Qt licenses if you want to use your products
## commercially, I recommend reading them on the official website:
## https://doc.qt.io/qtforpython/licenses.html
##
################################################################################

import sys
import os
import platform
from PyQt5.QtCore import pyqtSignal as pyqtss
from PyQt5.QtCore import QThread as qth 
from PyQt5.QtCore import QObject as qobj 

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent, QThread, )
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient, QMovie)
from PySide2.QtWidgets import *
from PIL import Image 
from PIL.ImageQt import ImageQt
from matplotlib import pyplot as plt
import shutil
import json
# GUI FILE
from app_modules import *

class TryOnWorker(qobj):
    finished = pyqtss()
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.try_on_img_path = ''
        self.try_on_img_pix_map = None

    def run(self):
        os.system('python Try-On/Run.py')
        self.try_on_img_path = 'output_image_generator/try-on.png'
        self.try_on_img_pix_map = QtGui.QPixmap(self.try_on_img_path)
        # resize image
        self.try_on_img_pix_map = self.try_on_img_pix_map.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.window.ui.generated_img_lbl.setPixmap(self.try_on_img_pix_map)
        self.finished.emit()
        

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ## PRINT ==> SYSTEM
        print('System: ' + platform.system())
        print('Version: ' +platform.release())

        ########################################################################
        ## START - WINDOW ATTRIBUTES
        ########################################################################

        ## REMOVE ==> STANDARD TITLE BAR
        UIFunctions.removeTitleBar(True)
        ## ==> END ##

        ## SET ==> WINDOW TITLE
        self.setWindowTitle('Main Window - Python Base')
        UIFunctions.labelTitle(self, 'Main Window - Python Base')
        UIFunctions.labelDescription(self, 'Set text')
        ## ==> END ##

        ## WINDOW SIZE ==> DEFAULT SIZE
        startSize = QSize(1000, 720)
        self.resize(startSize)
        self.setMinimumSize(startSize)
        # UIFunctions.enableMaximumSize(self, 500, 720)
        ## ==> END ##

        ## ==> CREATE MENUS
        ########################################################################

        ## ==> TOGGLE MENU SIZE
        self.ui.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))
        ## ==> END ##

        ## ==> ADD CUSTOM MENUS
        self.ui.stackedWidget.setMinimumWidth(20)
        UIFunctions.addNewMenu(self, "HOME", "btn_home", "url(:/16x16/icons/16x16/cil-home.png)", True)
        UIFunctions.addNewMenu(self, "Wardrobe", "btn_wardrobe", "url(:/16x16/icons/16x16/cil-user-follow.png)", True)
        UIFunctions.addNewMenu(self, "Add", "btn_add", "url(:/16x16/icons/16x16/cil-user-follow.png)", True)
        # add new button for Try On Module 
        UIFunctions.addNewMenu(self, "Try On", "btn_try_on", "Interface/icons/16x16/try_on.png", True, True)

        UIFunctions.addNewMenu(self, "Custom Widgets", "btn_widgets", "url(:/16x16/icons/16x16/cil-3d.png)", False)
        ## ==> END ##

        # START MENU => SELECTION
        UIFunctions.selectStandardMenu(self, "btn_home")
        ## ==> END ##

        ## ==> START PAGE
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
        ## ==> END ##

        ## USER ICON ==> SHOW HIDE
        UIFunctions.userIcon(self, "WM", "", True)
        ## ==> END ##


        ## ==> MOVE WINDOW / MAXIMIZE / RESTORE
        ########################################################################
        def moveWindow(event):
            # IF MAXIMIZED CHANGE TO NORMAL
            if UIFunctions.returStatus() == 1:
                UIFunctions.maximize_restore(self)

            # MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # WIDGET TO MOVE
        self.ui.frame_label_top_btns.mouseMoveEvent = moveWindow
        ## ==> END ##

        ## ==> LOAD DEFINITIONS
        ########################################################################
        UIFunctions.uiDefinitions(self)
        ## ==> END ##

        ########################################################################
        ## END - WINDOW ATTRIBUTES
        ############################## ---/--/--- ##############################




        ########################################################################
        #                                                                      #
        ## START -------------- WIDGETS FUNCTIONS/PARAMETERS ---------------- ##
        #                                                                      #
        ## ==> USER CODES BELLOW                                              ##
        ########################################################################



        ## ==> QTableWidget RARAMETERS
        ########################################################################
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        ## ==> END ##



        ########################################################################
        #                                                                      #
        ## END --------------- WIDGETS FUNCTIONS/PARAMETERS ----------------- ##
        #                                                                      #
        ############################## ---/--/--- ##############################


        ########################################################################
        #                                                                      #
        ## START --------------- TRY-ON FUNCTIONS----------------- ##############
        #                                                                      #
        ############################## ---/--/--- ##############################

        # person image 
        self.ui.person_im_try_on_btn.clicked.connect(lambda: self.open_image("person"))
        self.person_img_path = None
        self.person_img_pix_map = None
        # clothes image
        self.ui.cloth_im_try_on_btn.clicked.connect(lambda: self.open_image("cloth"))
        self.cloth_img_path = None
        self.cloth_img_pix_map = None
        # try on button
        self.ui.generated_im_try_on_btn.clicked.connect(self.try_on)
        self.try_on_img_path = None
        # binding wardrobe button
        self.ui.shirt_btn.clicked.connect(lambda: self.set_wardrobe_item(self.ui.shirt_btn))
        self.ui.pants_btn.clicked.connect(lambda: self.set_wardrobe_item(self.ui.pants_btn))
        self.ui.shoes_btn.clicked.connect(lambda: self.set_wardrobe_item(self.ui.shoes_btn))

        self.wardrobe_items = []

        self.set_icon()
        self.remove_border()
        ## SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##
    
    ########################################################################
    ## MENUS ==> DYNAMIC MENUS FUNCTIONS
    ########################################################################
    def Button(self):
        # GET BT CLICKED
        btnWidget = self.sender()

        # PAGE HOME
        if btnWidget.objectName() == "btn_home":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            UIFunctions.resetStyle(self, "btn_home")
            UIFunctions.labelPage(self, "Home")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE wardrobe
        if btnWidget.objectName() == "btn_wardrobe":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_wardrobe)
            UIFunctions.resetStyle(self, "btn_wardrobe")
            UIFunctions.labelPage(self, "Wardrobe")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

                # PAGE wardrobe
        if btnWidget.objectName() == "btn_add":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            UIFunctions.resetStyle(self, "btn_add")
            UIFunctions.labelPage(self, "Add")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))


        # PAGE WIDGETS
        if btnWidget.objectName() == "btn_widgets":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_widgets)
            UIFunctions.resetStyle(self, "btn_widgets")
            UIFunctions.labelPage(self, "Custom Widgets")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE TRY ON
        if btnWidget.objectName() == "btn_try_on":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_try_on)
            UIFunctions.resetStyle(self, "btn_try_on")
            UIFunctions.labelPage(self, "Try On")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

    ## ==> END ##

    ########################################################################
    ## START ==> APP EVENTS
    ########################################################################

    ## EVENT ==> MOUSE DOUBLE CLICK
    ########################################################################
    def eventFilter(self, watched, event):
        if watched == self.le and event.type() == QtCore.QEvent.MouseButtonDblClick:
            print("pos: ", event.pos())
    ## ==> END ##

    ## EVENT ==> MOUSE CLICK
    ########################################################################
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')
        if event.buttons() == Qt.MidButton:
            print('Mouse click: MIDDLE BUTTON')
    ## ==> END ##

    ## EVENT ==> KEY PRESSED
    ########################################################################
    def keyPressEvent(self, event):
        print('Key: ' + str(event.key()) + ' | Text Press: ' + str(event.text()))
    ## ==> END ##

    ## EVENT ==> RESIZE EVENT
    ########################################################################
    def resizeEvent(self, event):
        self.resizeFunction()
        return super(MainWindow, self).resizeEvent(event)

    def resizeFunction(self):
        print('Height: ' + str(self.height()) + ' | Width: ' + str(self.width()))
    ## ==> END ##

    ########################################################################
    ## END ==> APP EVENTS
    ############################## ---/--/--- ##############################



    ########################################################################
    ## User Defined functions
    ########################################################################
    def open_image(self, img_type):
        # get current directory
        if img_type == "person":
            self.person_img_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "G:/college/GP/try on/StyleAi/Try-On/data/train/image", "Image files (*.jpg *.gif *.png)")
            self.person_img_path  = self.person_img_path [0]
            self.person_img_pix_map = QtGui.QPixmap(self.person_img_path)
            # resize image
            self.person_img_pix_map = self.person_img_pix_map.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
            self.ui.person_img_lbl.setPixmap(self.person_img_pix_map)
        else:
            self.cloth_img_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "G:/college/GP/try on/StyleAi/Try-On/data/train/cloth", "Image files (*.jpg *.gif)")
            self.cloth_img_path = self.cloth_img_path[0]
            self.cloth_img_pix_map = QtGui.QPixmap(self.cloth_img_path)
            # resize image
            self.cloth_img_pix_map = self.cloth_img_pix_map.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
            self.ui.cloth_img_lbl.setPixmap(self.cloth_img_pix_map)
            
        
    def try_on(self):
        if self.person_img_path == None or self.cloth_img_path == None:
            pass
        else:
            # Delete all images in input and output folders
            for filename in os.listdir('Try-On/InputImages'):
                os.remove('Try-On/InputImages/' + filename)
            for filename in os.listdir('Try-On/InputClothesImages'):
                os.remove('Try-On/InputClothesImages/' + filename)
            for filename in os.listdir('output_image_generator'):
                os.remove('output_image_generator/' + filename)
            #Copy images to input and output folders
            shutil.copy(self.person_img_path, 'Try-On/InputImages')
            shutil.copy(self.cloth_img_path, 'Try-On/InputClothesImages')
            
            self.try_on_img_path = 'Interface/icons/load3.gif'
            self.movie = QtGui.QMovie(self.try_on_img_path)
            # resize movie
            self.movie.setScaledSize(QtCore.QSize(400, 300))
            self.ui.generated_img_lbl.setMovie(self.movie)
            self.movie.start()
            
            #Run the image generator
            self.thread = qth()
            self.worker = TryOnWorker(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
    def set_icon(self):
        shirt_icon = QIcon('Interface/icons/400x400/shirt.png')
        self.ui.shirt_btn.setIcon(shirt_icon)
        self.ui.shirt_btn.setIconSize(QtCore.QSize(400, 400))
        pants_icon = QIcon('Interface/icons/400x400/pants.png')
        self.ui.pants_btn.setIcon(pants_icon)
        self.ui.pants_btn.setIconSize(QtCore.QSize(400, 400))
        shoes_icon = QIcon('Interface/icons/400x400/shoes.png')
        self.ui.shoes_btn.setIcon(shoes_icon)
        self.ui.shoes_btn.setIconSize(QtCore.QSize(400, 400))

    def set_wardrobe_item(self, btnWidget):
        path = None
        page = None
        if btnWidget.objectName() == "shirt_btn":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_items)
            UIFunctions.resetStyle(self, "shirt_btn")
            UIFunctions.labelPage(self, "Shirts")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))
            path = "Interface/shirts"
            page = self.ui.scrollAreaWidgetContents_items
        if btnWidget.objectName() == "pants_btn":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_items)
            UIFunctions.resetStyle(self, "pants_btn")
            UIFunctions.labelPage(self, "Pants")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))
            path = "Interface/pants"
            page = self.ui.scrollAreaWidgetContents_items
        if btnWidget.objectName() == "shoes_btn":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_items)
            UIFunctions.resetStyle(self, "shoes_btn")
            UIFunctions.labelPage(self, "Shoes")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))
            path = "Interface/shoes"
            page = self.ui.scrollAreaWidgetContents_items

        if path == None or page == None:
            return
        
        self.get_wardrobe_item_img(path, page)


    def get_wardrobe_item_img(self, path, page):
        self.ui.article_lbl.setText("Article :")
        self.ui.color_lbl.setText("Color :")
        self.ui.gender_lbl.setText("Gender :")
        self.ui.usage_lbl.setText("Usage :")
        # remove pixmap from article_img_lbl
        self.ui.selected_img_lbl.setPixmap(QtGui.QPixmap(""))
        # make wardrobe_items empty
        if len(self.wardrobe_items) > 0:
            for btn in self.wardrobe_items:
                page.layout().removeWidget(btn)
                btn.deleteLater()
            self.wardrobe_items = []
            layout = page.layout()
        # Add buttons to the page
        else:
            layout = QtWidgets.QGridLayout(page)
        row = 0
        col = 0
        layout.setVerticalSpacing(100)
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                btn = QtWidgets.QPushButton()
                btn.setObjectName(filename)
                btn.setStyleSheet("border: none;")
                btn.setFixedSize(200, 200)
                btn.setIcon(QtGui.QIcon(path + "/" + filename))
                btn.setIconSize(QtCore.QSize(200, 200))
                self.wardrobe_items.append(btn)
                layout.addWidget(btn, row, col)
                col += 1
                if col == 4:
                    col = 0
                    row += 1
                btn.clicked.connect(lambda path=path, filename=filename: self.set_wardrobe_item_img(path, filename))
                
    def set_wardrobe_item_img(self, path, filename):
        pixmap = QtGui.QPixmap(path + "/" + filename)
        pixmap = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.ui.selected_img_lbl.setPixmap(pixmap)
        # set shirt_selected_img_lbl to the image of the button
        if path == "Interface/shirts":
            json_file = open('Interface/shirtLabels/' + filename[:-4] + '.json')
        elif path == "Interface/pants":
            json_file = open('Interface/pantLabels/' + filename[:-4] + '.json')
        elif path == "Interface/shoes":
            json_file = open('Interface/shoeLabels/' + filename[:-4] + '.json')
        json_data = json.load(json_file)
        self.ui.article_lbl.setText("Article :" + json_data['Article'])
        self.ui.color_lbl.setText("Color :" + json_data['Color'])
        self.ui.gender_lbl.setText("Gender :" + json_data['Gender'])
        self.ui.usage_lbl.setText("Usage :" + json_data['Usage'])
    def remove_border(self):
        # Remove border from scrollAreaWidgetContents_shirts
        self.ui.scrollArea_items.setStyleSheet("border: none;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeui.ttf')
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeuib.ttf')
    window = MainWindow()
    sys.exit(app.exec_())
