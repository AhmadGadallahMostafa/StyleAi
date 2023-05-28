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
import time
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
        self.window.tryonMovie.stop()
        self.try_on_img_path = 'output_image_generator/try-on.png'
        self.try_on_img_pix_map = QtGui.QPixmap(self.try_on_img_path)
        # resize image
        self.try_on_img_pix_map = self.try_on_img_pix_map.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.window.ui.generated_img_lbl.setPixmap(self.try_on_img_pix_map)

        self.window.ui.generated_im_try_on_btn.setEnabled(True)
        
        self.finished.emit()

class ClassifierWorker(qobj):
    finished = pyqtss()
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.classifier_img_top_path = ''
        self.classifier_img_bottom_path = ''
        self.classifier_img_shoes_path = ''
        self.classifier_img_top_pix_map = None
        self.classifier_img_bottom_pix_map = None
        self.classifier_img_shoes_pix_map = None

    def run(self):
        # sleep for 5 seconds
        os.system('python Segmentation/segment.py')
        os.system('python Classification\Run.py')
        self.window.classifierMovie.stop()

        # emit signal to update the UI
        self.classifier_img_top_path = 'Interface/classifieroutput/top.jpg'
        self.classifier_img_bottom_path = 'Interface/classifieroutput/bottom.jpg'
        self.classifier_img_shoes_path = 'Interface/classifieroutput/full_body.jpg'
        self.classifier_img_top_pix_map = QtGui.QPixmap(self.classifier_img_top_path)
        self.classifier_img_bottom_pix_map = QtGui.QPixmap(self.classifier_img_bottom_path)
        self.classifier_img_shoes_pix_map = QtGui.QPixmap(self.classifier_img_shoes_path)
        # resize image
        self.classifier_img_top_pix_map = self.classifier_img_top_pix_map.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.classifier_img_bottom_pix_map = self.classifier_img_bottom_pix_map.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.classifier_img_shoes_pix_map = self.classifier_img_shoes_pix_map.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
        self.window.ui.classifier_top_lbl.setPixmap(self.classifier_img_top_pix_map)
        self.window.ui.classifier_bottom_lbl.setPixmap(self.classifier_img_bottom_pix_map)
        self.window.ui.classifier_shoes_lbl.setPixmap(self.classifier_img_shoes_pix_map)
        # get the json files
        json_file_top = open('Interface/classifieroutput/top.json')
        json_data_top = json.load(json_file_top)
        json_file_bottom = open('Interface/classifieroutput/bottom.json')
        json_data_bottom = json.load(json_file_bottom)
        json_file_shoes = open('Interface/classifieroutput/full_body.json')
        json_data_shoes = json.load(json_file_shoes)

        self.window.ui.article_classifier_top_lbl.setText("Article :" + json_data_top['Article'])
        self.window.ui.color_classifier_top_lbl.setText("Color :" + json_data_top['Color'])
        self.window.ui.gender_classifier_top_lbl.setText("Gender :" + json_data_top['Gender'])
        self.window.ui.usage_classifier_top_lbl.setText("Usage :" + json_data_top['Usage'])

        self.window.ui.article_classifier_bottom_lbl.setText("Article :" + json_data_bottom['Article'])
        self.window.ui.color_classifier_bottom_lbl.setText("Color :" + json_data_bottom['Color'])
        self.window.ui.gender_classifier_bottom_lbl.setText("Gender :" + json_data_bottom['Gender'])
        self.window.ui.usage_classifier_bottom_lbl.setText("Usage :" + json_data_bottom['Usage'])

        self.window.ui.article_classifier_shoes_lbl.setText("Article :" + json_data_shoes['Article'])
        self.window.ui.color_classifier_shoes_lbl.setText("Color :" + json_data_shoes['Color'])
        self.window.ui.gender_classifier_shoes_lbl.setText("Gender :" + json_data_shoes['Gender'])
        self.window.ui.usage_classifier_shoes_lbl.setText("Usage :" + json_data_shoes['Usage'])

        self.window.ui.classify_input_btn.setEnabled(True)

        self.window.ui.classify_save_top_btn.setVisible(True)
        self.window.ui.classify_save_bottom_btn.setVisible(True)
        self.window.ui.classify_save_shoes_btn.setVisible(True)
        self.window.ui.classify_save_top_btn.setEnabled(True)
        self.window.ui.classify_save_bottom_btn.setEnabled(True)
        self.window.ui.classify_save_shoes_btn.setEnabled(True)
        self.finished.emit()

class RecommenderScoreWorker(qobj):
    finished = pyqtss()
    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        # sleep for 5 seconds
        os.system('python Recommendation\score.py')
        self.window.recommenderScoreMovie.stop()
        # read score from Interface\score.txt
        score_file = open('Interface/score.txt', 'r')
        score = score_file.read()
        # round score to 4 decimal places
        score = round(float(score), 4)
        self.window.ui.recommender_outfit_score_lbl.setText("Score : " + str(score))
        self.window.ui.get_score_btn.setEnabled(True)

        self.finished.emit()

class RecommenderImprovedOutfitWorker(qobj):
    finished = pyqtss()
    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        # sleep for 5 seconds
        os.system('python Recommendation\score.py')
        self.window.recommenderImprovedOutfitMovie.stop()
        self.window.recommenderImprovedScoreMovie.stop()
        
        # emit signal to update the UI
        self.classifier_improved_img_top_path = 'Interface\outfit_evaluated/top.jpg'
        self.classifier_improved_img_bottom_path = 'Interface\outfit_evaluated/bottom.jpg'
        self.classifier_improved_img_shoes_path = 'Interface\outfit_evaluated/shoes.jpg'
        self.classifier_improved_img_top_pix_map = QtGui.QPixmap(self.classifier_improved_img_top_path)
        self.classifier_improved_img_bottom_pix_map = QtGui.QPixmap(self.classifier_improved_img_bottom_path)
        self.classifier_improved_img_shoes_pix_map = QtGui.QPixmap(self.classifier_improved_img_shoes_path)

        # read json of improved article
        json_data_best = json.load(open('Interface/best_img_path.json'))
        # get the image paths
        # check if top key exists
        if 'top' in json_data_best.keys():
            self.classifier_improved_img_top_path = json_data_best['top']
            self.classifier_improved_img_top_pix_map = QtGui.QPixmap(self.classifier_improved_img_top_path)
        if 'bottom' in json_data_best.keys():
            self.classifier_improved_img_bottom_path = json_data_best['bottom']
            self.classifier_improved_img_bottom_pix_map = QtGui.QPixmap(self.classifier_improved_img_bottom_path)
        if 'shoes' in json_data_best.keys():
            self.classifier_improved_img_shoes_path = json_data_best['shoes']
            self.classifier_improved_img_shoes_pix_map = QtGui.QPixmap(self.classifier_improved_img_shoes_path)

        # resize image
        self.classifier_improved_img_top_pix_map = self.classifier_improved_img_top_pix_map.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
        self.classifier_improved_img_bottom_pix_map = self.classifier_improved_img_bottom_pix_map.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
        self.classifier_improved_img_shoes_pix_map = self.classifier_improved_img_shoes_pix_map.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
        self.window.ui.top_recommender_improved_outfit_lbl.setPixmap(self.classifier_improved_img_top_pix_map)
        self.window.ui.bottom_recommender_improved_outfit_lbl.setPixmap(self.classifier_improved_img_bottom_pix_map)
        self.window.ui.shoes_recommender_improved_outfit_lbl.setPixmap(self.classifier_improved_img_shoes_pix_map)

        score_file = open('Interface/best_score.txt', 'r')
        score = score_file.read()
        # round score to 4 decimal places
        score = round(float(score), 4)
        # get score
        self.window.ui.recommender_improved_outfit_score_lbl.setText("Score :" + str(score))
        
        self.window.ui.improve_outfit_btn.setEnabled(True)
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
        UIFunctions.addNewMenu(self, "Get Outfit", "btn_get_outfit", "url(:/16x16/icons/16x16/cil-user-follow.png)", True)
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
        # classifier image
        self.ui.add_classifier_input_btn.clicked.connect(lambda: self.open_image("classifier"))
        self.classifier_img_path = None
        self.classifier_img_pix_map = None
        # top image
        self.ui.top_recommender_manual_get_outfit_btn.clicked.connect(lambda: self.open_image("top"))
        self.recommender_top_path = None
        self.recommender_top_pix_map = None
        # bottom image
        self.ui.bottom_recommender_manual_get_outfit_btn.clicked.connect(lambda: self.open_image("bottom"))
        self.recommender_bottom_path = None
        self.recommender_bottom_pix_map = None
        # shoes image
        self.ui.shoes_recommender_manual_get_outfit_btn.clicked.connect(lambda: self.open_image("shoes"))
        self.recommender_shoes_path = None
        self.recommender_shoes_pix_map = None

        # classifier button
        self.ui.classify_input_btn.clicked.connect(self.classify)
        # try on button
        self.ui.generated_im_try_on_btn.clicked.connect(self.try_on)
        self.try_on_img_path = None
        # binding wardrobe button
        self.ui.shirt_btn.clicked.connect(lambda: self.set_wardrobe_item(self.ui.shirt_btn))
        self.ui.pants_btn.clicked.connect(lambda: self.set_wardrobe_item(self.ui.pants_btn))
        self.ui.shoes_btn.clicked.connect(lambda: self.set_wardrobe_item(self.ui.shoes_btn))
        # binding save button
        self.ui.classify_save_top_btn.clicked.connect(lambda: self.save_classifier_output("top"))
        self.ui.classify_save_bottom_btn.clicked.connect(lambda: self.save_classifier_output("bottom"))
        self.ui.classify_save_shoes_btn.clicked.connect(lambda: self.save_classifier_output("full_body"))
        # binding recommender buttons
        self.ui.top_recommender_get_outfit_btn.clicked.connect(lambda: self.set_recommender_item("top"))
        self.ui.bottom_recommender_get_outfit_btn.clicked.connect(lambda: self.set_recommender_item("bottom"))
        self.ui.shoes_recommender_get_outfit_btn.clicked.connect(lambda: self.set_recommender_item("shoes"))
        # binding recommender score button
        self.ui.get_score_btn.clicked.connect(self.get_score_recommender)
        # binding improve outfit button
        self.ui.improve_outfit_btn.clicked.connect(self.get_improved_outfit)


        # make save buttons invisible and disabled
        self.ui.classify_save_top_btn.setVisible(False)
        self.ui.classify_save_bottom_btn.setVisible(False)
        self.ui.classify_save_shoes_btn.setVisible(False)
        self.ui.classify_save_top_btn.setEnabled(False)
        self.ui.classify_save_bottom_btn.setEnabled(False)
        self.ui.classify_save_shoes_btn.setEnabled(False)

        self.wardrobe_items = []
        self.recommender_items = []
        

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

        # PAGE add
        if btnWidget.objectName() == "btn_add":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_add)
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

        if btnWidget.objectName() == "btn_get_outfit":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_recommender_get_outfit)
            UIFunctions.resetStyle(self, "btn_get_outfit")
            UIFunctions.labelPage(self, "Get Outfit")
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
        elif img_type == "cloth":
            self.cloth_img_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "G:/college/GP/try on/StyleAi/Try-On/data/train/cloth", "Image files (*.jpg *.gif)")
            self.cloth_img_path = self.cloth_img_path[0]
            self.cloth_img_pix_map = QtGui.QPixmap(self.cloth_img_path)
            # resize image
            self.cloth_img_pix_map = self.cloth_img_pix_map.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
            self.ui.cloth_img_lbl.setPixmap(self.cloth_img_pix_map)
        elif img_type == "classifier":
            self.input_classifier_img_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "G:/college/GP/try on/StyleAi/Try-On/data/train/cloth", "Image files (*.jpg *.gif *.png)")
            self.input_classifier_img_path = self.input_classifier_img_path[0]
            self.input_classifier_img_pix_map = QtGui.QPixmap(self.input_classifier_img_path)
            # resize image
            self.input_classifier_img_pix_map = self.input_classifier_img_pix_map.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
            self.ui.classifier_input_lbl.setPixmap(self.input_classifier_img_pix_map)
        elif img_type == "top":
            self.recommender_top_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "Interface\shirts", "Image files (*.jpg *.gif *.png)")
            self.recommender_top_path = self.recommender_top_path[0]
            self.recommender_top_pix_map = QtGui.QPixmap(self.recommender_top_path)
            # resize image
            self.recommender_top_pix_map = self.recommender_top_pix_map.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
            self.ui.top_recommender_outfit_lbl.setPixmap(self.recommender_top_pix_map)
        elif img_type == "bottom":
            self.recommender_bottom_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "Interface\pants", "Image files (*.jpg *.gif *.png)")
            self.recommender_bottom_path = self.recommender_bottom_path[0]
            self.recommender_bottom_pix_map = QtGui.QPixmap(self.recommender_bottom_path)
            # resize image
            self.recommender_bottom_pix_map = self.recommender_bottom_pix_map.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
            self.ui.bottom_recommender_outfit_lbl.setPixmap(self.recommender_bottom_pix_map)
        elif img_type == "shoes":
            self.recommender_shoes_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "Interface\shoes", "Image files (*.jpg *.gif *.png)")
            self.recommender_shoes_path = self.recommender_shoes_path[0]
            self.recommender_shoes_pix_map = QtGui.QPixmap(self.recommender_shoes_path)
            # resize image
            self.recommender_shoes_pix_map = self.recommender_shoes_pix_map.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
            self.ui.shoes_recommender_outfit_lbl.setPixmap(self.recommender_shoes_pix_map)
        
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
            self.tryonMovie = QtGui.QMovie(self.try_on_img_path)
            # resize movie
            self.tryonMovie.setScaledSize(QtCore.QSize(400, 300))
            self.ui.generated_img_lbl.setMovie(self.tryonMovie)
            self.tryonMovie.start()
            
            #self.ui.classify_input_btn.setEnabled(False)
            self.ui.generated_im_try_on_btn.setEnabled(False)
            #self.ui.get_score_btn.setEnabled(False)

            #Run the image generator
            self.tryonThread = qth()
            self.tryonWorker = TryOnWorker(self)
            self.tryonWorker.moveToThread(self.tryonThread)
            self.tryonThread.started.connect(self.tryonWorker.run)
            self.tryonWorker.finished.connect(self.tryonThread.quit)
            self.tryonWorker.finished.connect(self.tryonWorker.deleteLater)
            self.tryonThread.finished.connect(self.tryonThread.deleteLater)
            self.tryonThread.start()

    def classify(self):
        if self.input_classifier_img_path == None:
            pass
        else:
            # Delete all images in input and output folders
            for filename in os.listdir('Interface\input_segmentation'):
                os.remove('Interface\input_segmentation/' + filename)

            for filename in os.listdir('Interface\classifieroutput'):
                os.remove('Interface\classifieroutput/' + filename)

            for filename in os.listdir('Interface\output_segmentation'):
                os.remove('Interface\output_segmentation/' + filename)
            #Copy images to input and output folders
            shutil.copy(self.input_classifier_img_path, 'Interface\input_segmentation')
            
            self.classifier_img_path = 'Interface/icons/load3.gif'
            self.classifierMovie = QtGui.QMovie('Interface/icons/load3.gif')
            # resize movie
            self.classifierMovie.setScaledSize(QtCore.QSize(400, 300))
            self.ui.classifier_top_lbl.setMovie(self.classifierMovie)
            self.ui.classifier_bottom_lbl.setMovie(self.classifierMovie)
            self.ui.classifier_shoes_lbl.setMovie(self.classifierMovie)
            self.classifierMovie.setSpeed(100)
            self.classifierMovie.start()
            
            self.ui.article_classifier_top_lbl.setText("Article :")
            self.ui.color_classifier_top_lbl.setText("Color :")
            self.ui.gender_classifier_top_lbl.setText("Gender :")
            self.ui.usage_classifier_top_lbl.setText("Usage :")

            self.ui.article_classifier_bottom_lbl.setText("Article :")
            self.ui.color_classifier_bottom_lbl.setText("Color :")
            self.ui.gender_classifier_bottom_lbl.setText("Gender :")
            self.ui.usage_classifier_bottom_lbl.setText("Usage :")

            self.ui.article_classifier_shoes_lbl.setText("Article :")
            self.ui.color_classifier_shoes_lbl.setText("Color :")
            self.ui.gender_classifier_shoes_lbl.setText("Gender :")
            self.ui.usage_classifier_shoes_lbl.setText("Usage :")

            self.ui.classify_input_btn.setEnabled(False)
            #self.ui.generated_im_try_on_btn.setEnabled(False)
            #self.ui.get_score_btn.setEnabled(False)

            #Run the classifier
            self.classifierThread = qth()
            self.classifierWorker = ClassifierWorker(self)
            self.classifierWorker.moveToThread(self.classifierThread)
            self.classifierThread.started.connect(self.classifierWorker.run)
            self.classifierWorker.finished.connect(self.classifierThread.quit)
            self.classifierWorker.finished.connect(self.classifierWorker.deleteLater)
            self.classifierThread.start()

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

        self.ui.top_recommender_get_outfit_btn.setIcon(shirt_icon)
        self.ui.top_recommender_get_outfit_btn.setIconSize(QtCore.QSize(150, 150))
        self.ui.bottom_recommender_get_outfit_btn.setIcon(pants_icon)
        self.ui.bottom_recommender_get_outfit_btn.setIconSize(QtCore.QSize(150, 150))
        self.ui.shoes_recommender_get_outfit_btn.setIcon(shoes_icon)
        self.ui.shoes_recommender_get_outfit_btn.setIconSize(QtCore.QSize(150, 150))

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
        # Remove border from scrollArea_items
        self.ui.scrollArea_items.setStyleSheet("border: none;")
        # Remove border from scrollArea_recommender_items
        self.ui.scrollArea_recommender_items.setStyleSheet("border: none;")

    def save_classifier_output(self, output):
        num = 0
        img_path = ''
        json_path = ''
        if output == "top":
            num = len(os.listdir("Interface/shirts")) + 1
            img_path = 'Interface/classifieroutput/top.jpg'
            json_path = 'Interface/classifieroutput/top.json'
            # save in shirts folder and shirtLabels folder and change name to num.jpg
            shutil.copy(img_path, 'Interface/shirts/' + str(num) + '.jpg')
            shutil.copy(json_path, 'Interface/shirtLabels/' + str(num) + '.json')
            self.ui.classify_save_top_btn.setEnabled(False)
            self.ui.classify_save_top_btn.setVisible(False)            
        elif output == "bottom":
            num = len(os.listdir("Interface/pants")) + 1
            img_path = 'Interface/classifieroutput/bottom.jpg'
            json_path = 'Interface/classifieroutput/bottom.json'
            # save in pants folder and pantLabels folder and change name to num.jpg
            shutil.copy(img_path, 'Interface/pants/' + str(num) + '.jpg')
            shutil.copy(json_path, 'Interface/pantLabels/' + str(num) + '.json')
            self.ui.classify_save_bottom_btn.setVisible(False)
            self.ui.classify_save_bottom_btn.setEnabled(False)
        elif output == "full_body":
            num = len(os.listdir("Interface/pants")) + 1
            img_path = 'Interface/classifieroutput/full_body.jpg'
            json_path = 'Interface/classifieroutput/full_body.json'
            # save in shoes folder and shoeLabels folder and change name to num.jpg
            shutil.copy(img_path, 'Interface/pants/' + str(num) + '.jpg')
            shutil.copy(json_path, 'Interface/pantLabels/' + str(num) + '.json')
            self.ui.classify_save_shoes_btn.setVisible(False)
            self.ui.classify_save_shoes_btn.setEnabled(False)

    def set_recommender_item(self, inputType):
        path = None
        page = None
        if inputType == "top":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_recommender_items)
            UIFunctions.labelPage(self, "Tops")
            path = "Interface/shirts"
            page = self.ui.scrollAreaWidgetContents_recommender_items 
        elif inputType == "bottom":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_recommender_items)
            UIFunctions.labelPage(self, "Bottoms")
            path = "Interface/pants"
            page = self.ui.scrollAreaWidgetContents_recommender_items
        elif inputType == "shoes":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_recommender_items)
            UIFunctions.labelPage(self, "Shoes")
            path = "Interface/shoes"
            page = self.ui.scrollAreaWidgetContents_recommender_items
        
        if path == None or page == None:
            return
        
        self.get_recommender_item(path, page, inputType)

    def get_recommender_item(self, path, page, inputType):
        if self.recommender_items:
            for btn in self.recommender_items:
                page.layout().removeWidget(btn)
                btn.deleteLater()
            self.recommender_items = []
            layout = page.layout()
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
                self.recommender_items.append(btn)
                layout.addWidget(btn, row, col)
                col += 1
                if col == 5:
                    col = 0
                    row += 1
                btn.clicked.connect(lambda path=path, filename=filename, inputType=inputType: self.set_recommender_item_img(path, filename, inputType))

    def set_recommender_item_img(self, path, filename, inputType):
        pixmap = QtGui.QPixmap(path + "/" + filename)
        pixmap = pixmap.scaled(250, 250, QtCore.Qt.KeepAspectRatio)
        if inputType == "top":
            self.ui.top_recommender_outfit_lbl.setPixmap(pixmap)
            self.recommender_top_path = path + "/" + filename
        elif inputType == "bottom":
            self.ui.bottom_recommender_outfit_lbl.setPixmap(pixmap)
            self.recommender_bottom_path = path + "/" + filename
        elif inputType == "shoes":
            self.ui.shoes_recommender_outfit_lbl.setPixmap(pixmap)
            self.recommender_shoes_path = path + "/" + filename
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_recommender_get_outfit)

    def get_score_recommender(self):
        if self.recommender_top_path == None or self.recommender_bottom_path == None or self.recommender_shoes_path == None:
            pass
        else:
            # Delete previous outfit
            # for filename in os.listdir('Try-On/InputImages'):
            #     os.remove('Try-On/InputImages/' + filename)
            # for filename in os.listdir('Try-On/InputClothesImages'):
            #     os.remove('Try-On/InputClothesImages/' + filename)
            # for filename in os.listdir('output_image_generator'):
            #     os.remove('output_image_generator/' + filename)
            shutil.copy(self.recommender_top_path, 'Interface\outfit_evaluated/top.jpg')
            shutil.copy(self.recommender_bottom_path, 'Interface\outfit_evaluated/bottom.jpg')
            shutil.copy(self.recommender_shoes_path, 'Interface\outfit_evaluated/shoes.jpg')

            self.ui.recommender_outfit_score_lbl.setText("")
            self.recommenderScoreMovie = QtGui.QMovie('Interface/icons/load3.gif')
            # resize movie
            self.recommenderScoreMovie.setScaledSize(QtCore.QSize(150, 150))
            self.ui.recommender_outfit_score_lbl.setMovie(self.recommenderScoreMovie)
            self.recommenderScoreMovie.setSpeed(100)
            self.recommenderScoreMovie.start()

            self.ui.get_score_btn.setEnabled(False)

            # Run recommender score
            self.recommenderScoreThread = qth()
            self.recommenderWorker = RecommenderScoreWorker(self)
            self.recommenderWorker.moveToThread(self.recommenderScoreThread)
            self.recommenderScoreThread.started.connect(self.recommenderWorker.run)
            self.recommenderWorker.finished.connect(self.recommenderScoreThread.quit)
            self.recommenderWorker.finished.connect(self.recommenderScoreThread.deleteLater)
            self.recommenderScoreThread.start()

    def get_improved_outfit(self):
        if self.recommender_top_path == None or self.recommender_bottom_path == None or self.recommender_shoes_path == None:
            pass
        else:
            # Delete previous outfit
            # for filename in os.listdir('Try-On/InputImages'):
            #     os.remove('Try-On/InputImages/' + filename)
            # for filename in os.listdir('Try-On/InputClothesImages'):
            #     os.remove('Try-On/InputClothesImages/' + filename)
            # for filename in os.listdir('output_image_generator'):
            #     os.remove('output_image_generator/' + filename)
            # shutil.copy(self.recommender_top_path, 'Interface\outfit_evaluated/top.jpg')
            # shutil.copy(self.recommender_bottom_path, 'Interface\outfit_evaluated/bottom.jpg')
            # shutil.copy(self.recommender_shoes_path, 'Interface\outfit_evaluated/shoes.jpg')
            self.recommenderImprovedOutfitMovie = QtGui.QMovie('Interface/icons/load3.gif')
            # resize movie
            self.recommenderImprovedOutfitMovie.setScaledSize(QtCore.QSize(300, 300))
            self.ui.top_recommender_improved_outfit_lbl.setMovie(self.recommenderImprovedOutfitMovie)
            self.ui.bottom_recommender_improved_outfit_lbl.setMovie(self.recommenderImprovedOutfitMovie)
            self.ui.shoes_recommender_improved_outfit_lbl.setMovie(self.recommenderImprovedOutfitMovie)
            self.recommenderImprovedOutfitMovie.setSpeed(100)
            self.recommenderImprovedOutfitMovie.start()

            self.ui.recommender_improved_outfit_score_lbl.setText("")
            self.recommenderImprovedScoreMovie = QtGui.QMovie('Interface/icons/load3.gif')
            # resize movie
            self.recommenderImprovedScoreMovie.setScaledSize(QtCore.QSize(150, 150))
            self.ui.recommender_improved_outfit_score_lbl.setMovie(self.recommenderImprovedScoreMovie)
            self.recommenderImprovedScoreMovie.setSpeed(100)
            self.recommenderImprovedScoreMovie.start()
            
            self.ui.improve_outfit_btn.setEnabled(False)

            # Run recommender improved outfit
            self.recommenderImprovedOutfitThread = qth()
            self.recommenderImprovedOutfitWorker = RecommenderImprovedOutfitWorker(self)
            self.recommenderImprovedOutfitWorker.moveToThread(self.recommenderImprovedOutfitThread)
            self.recommenderImprovedOutfitThread.started.connect(self.recommenderImprovedOutfitWorker.run)
            self.recommenderImprovedOutfitWorker.finished.connect(self.recommenderImprovedOutfitThread.quit)
            self.recommenderImprovedOutfitWorker.finished.connect(self.recommenderImprovedOutfitThread.deleteLater)
            self.recommenderImprovedOutfitThread.start()
            




if __name__ == "__main__":
    app = QApplication(sys.argv)
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeui.ttf')
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeuib.ttf')
    window = MainWindow()
    sys.exit(app.exec_())
