from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from preprocessing import preprocess
from smarthouse import smarthouse
from utils import plot_classification_report, plot_confusion_matrix


class Ui_Dialog(object):
    def toggle_mode(self):
        self.days_spin.setEnabled(not self.days_spin.isEnabled())
        self.days_label.setEnabled(not self.days_label.isEnabled())
        self.samples_spin.setEnabled(not self.samples_spin.isEnabled())
        self.nsamples_label.setEnabled(not self.nsamples_label.isEnabled())

    def toggle_dataset(self):
        self.max_days = 13 if self.max_days == 20 else 20
        self.days_spin.setMaximum(self.max_days)

    def format_sequences(self, sample, predicted):
        sample = list(
            map(lambda v: f"&nbsp;&nbsp;{v}" if v < 10 else str(v), sample)
        )
        predicted = list(
            map(lambda v: f"&nbsp;&nbsp;{v}" if v < 10 else str(v), predicted)
        )

        for i in range(len(sample)):
            if sample[i] == predicted[i]:
                sample[i] = predicted[
                    i
                ] = f"<font face='mono' color='green'>&nbsp;{sample[i]}</font>"
            else:
                sample[
                    i
                ] = f"<font face='mono' color='red'>&nbsp;{sample[i]}</font>"
                predicted[
                    i
                ] = f"<font face='mono' color='red'>&nbsp;{predicted[i]}</font>"

        sample_rows = [
            " ".join(sample[x : x + 5]) for x in range(0, len(sample), 5)
        ]
        sample_text = "<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(sample_rows)
        sample_text = "&nbsp;&nbsp;&nbsp;&nbsp;" + sample_text

        predicted_rows = [
            " ".join(predicted[x : x + 5]) for x in range(0, len(predicted), 5)
        ]
        predicted_text = "<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(predicted_rows)
        predicted_text = "&nbsp;&nbsp;&nbsp;&nbsp;" + predicted_text

        return sample_text, predicted_text

    @QtCore.pyqtSlot()
    def do_process(self):
        def update_progress_bar(value, bar):
            if bar == "A":
                self.process_progress1.setValue(value)
            else:
                self.process_progress2.setValue(value)
            QtWidgets.QApplication.processEvents()

        parameters = {}
        parameters["length"] = self.slice_spinbox.value()
        parameters["use_day_period"] = self.period_checkbox.isChecked()
        parameters["onupdate"] = update_progress_bar
        parameters["on_att"] = "id"
        if location_only:
            parameters["on_att"] = "location"
        if place_only:
            paramenters["on_att"] = "place"

        self.run_button.setEnabled(False)
        self.process_progress1.setValue(0)
        self.process_progress2.setValue(0)
        preprocess(**parameters)
        self.run_button.setEnabled(True)

    def do_viterbi(self):
        parameters = {}
        parameters["dataset"] = "A" if self.a_radio.isChecked() else "B"
        if self.split_radio.isChecked():
            parameters["test_days"] = self.days_spin.value()

        if self.sampling_radio.isChecked():
            parameters["n_samples"] = self.samples_spin.value()

        sample, predicted, accuracy = smarthouse(**parameters)

        plot_classification_report(sample, predicted)
        plt.figure(2)
        plot_confusion_matrix(
            sample, predicted, list(map(str, range(max(sample) + 1)))
        )

        sample_text, predicted_text = self.format_sequences(sample, predicted)

        self.accuracy_value_label.setText(f"{accuracy*100:.3f}")
        self.sample_textbrowser.setText(sample_text)
        self.predicted_textbrowser.setText(predicted_text)

        plt.show()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(480, 640)

        # Processing
        self.processing_groupbox = QtWidgets.QGroupBox(Dialog)
        self.processing_groupbox.setGeometry(QtCore.QRect(10, 10, 461, 121))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.processing_groupbox.setFont(font)
        self.processing_groupbox.setObjectName("processing_groupbox")

        # Timeslice
        self.slice_spinbox = QtWidgets.QSpinBox(self.processing_groupbox)
        self.slice_spinbox.setGeometry(QtCore.QRect(110, 50, 71, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.slice_spinbox.setFont(font)
        self.slice_spinbox.setAlignment(
            QtCore.Qt.AlignRight
            | QtCore.Qt.AlignTrailing
            | QtCore.Qt.AlignVCenter
        )
        self.slice_spinbox.setMaximum(10000)
        self.slice_spinbox.setProperty("value", 60)
        self.slice_spinbox.setObjectName("slice_spinbox")
        self.slice_label = QtWidgets.QLabel(self.processing_groupbox)
        self.slice_label.setGeometry(QtCore.QRect(20, 50, 91, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.slice_label.setFont(font)
        self.slice_label.setObjectName("slice_label")

        # Day Period
        self.period_checkbox = QtWidgets.QCheckBox(self.processing_groupbox)
        self.period_checkbox.setGeometry(QtCore.QRect(15, 80, 131, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.period_checkbox.setFont(font)
        self.period_checkbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.period_checkbox.setObjectName("period_checkbox")

        # Process Button
        self.process_button = QtWidgets.QPushButton(self.processing_groupbox)
        self.process_button.setGeometry(QtCore.QRect(340, 30, 113, 32))
        self.process_button.setObjectName("process_button")
        self.process_button.clicked.connect(self.do_process)

        # Location Only
        self.location_checkbox = QtWidgets.QCheckBox(self.processing_groupbox)
        self.location_checkbox.setGeometry(QtCore.QRect(190, 50, 131, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.location_checkbox.setFont(font)
        self.location_checkbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.location_checkbox.setObjectName("location_checkbox")

        # Place Only
        self.place_checkbox = QtWidgets.QCheckBox(self.processing_groupbox)
        self.place_checkbox.setGeometry(QtCore.QRect(190, 80, 131, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.place_checkbox.setFont(font)
        self.place_checkbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.place_checkbox.setObjectName("place_checkbox")

        # Progress Bar
        self.process_progress1 = QtWidgets.QProgressBar(
            self.processing_groupbox
        )
        self.process_progress1.setGeometry(QtCore.QRect(340, 60, 113, 32))
        self.process_progress2 = QtWidgets.QProgressBar(
            self.processing_groupbox
        )
        self.process_progress2.setGeometry(QtCore.QRect(340, 80, 113, 32))

        # Hidden Markov Model
        self.hmm_groupbox = QtWidgets.QGroupBox(Dialog)
        self.hmm_groupbox.setGeometry(QtCore.QRect(10, 150, 460, 161))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.hmm_groupbox.setFont(font)
        self.hmm_groupbox.setObjectName("hmm_groupbox")

        # Dataset
        self.dataset_groupbox = QtWidgets.QGroupBox(self.hmm_groupbox)
        self.dataset_groupbox.setGeometry(QtCore.QRect(10, 50, 71, 101))
        self.dataset_groupbox.setObjectName("dataset_groupbox")
        self.a_radio = QtWidgets.QRadioButton(self.dataset_groupbox)
        self.a_radio.setGeometry(QtCore.QRect(10, 30, 41, 20))
        self.a_radio.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.a_radio.setChecked(True)
        self.a_radio.setObjectName("a_radio")
        self.b_radio = QtWidgets.QRadioButton(self.dataset_groupbox)
        self.b_radio.setGeometry(QtCore.QRect(10, 60, 41, 20))
        self.b_radio.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.b_radio.setObjectName("b_radio")
        self.a_radio.toggled.connect(lambda x: self.toggle_dataset())

        # Mode
        self.mode_groupbox = QtWidgets.QGroupBox(self.hmm_groupbox)
        self.mode_groupbox.setGeometry(QtCore.QRect(100, 50, 351, 101))
        self.mode_groupbox.setObjectName("mode_groupbox")

        # Random Sampling
        self.sampling_radio = QtWidgets.QRadioButton(self.mode_groupbox)
        self.sampling_radio.setGeometry(QtCore.QRect(10, 30, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.sampling_radio.setFont(font)
        self.sampling_radio.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.sampling_radio.setObjectName("sampling_radio")
        self.sampling_radio.setChecked(True)

        # Split dataset
        self.split_radio = QtWidgets.QRadioButton(self.mode_groupbox)
        self.split_radio.setGeometry(QtCore.QRect(10, 60, 141, 26))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.split_radio.setFont(font)
        self.split_radio.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.split_radio.setObjectName("split_radio")
        self.split_radio.toggled.connect(lambda x: self.toggle_mode())

        # Sample number
        self.samples_spin = QtWidgets.QSpinBox(self.mode_groupbox)
        self.samples_spin.setGeometry(QtCore.QRect(250, 28, 81, 25))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.samples_spin.setFont(font)
        self.samples_spin.setAlignment(
            QtCore.Qt.AlignRight
            | QtCore.Qt.AlignTrailing
            | QtCore.Qt.AlignVCenter
        )
        self.samples_spin.setMaximum(100000)
        self.samples_spin.setProperty("value", 3000)
        self.samples_spin.setObjectName("samples_spin")
        self.nsamples_label = QtWidgets.QLabel(self.mode_groupbox)
        self.nsamples_label.setGeometry(QtCore.QRect(150, 30, 91, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.nsamples_label.setFont(font)
        self.nsamples_label.setAlignment(
            QtCore.Qt.AlignRight
            | QtCore.Qt.AlignTrailing
            | QtCore.Qt.AlignVCenter
        )
        self.nsamples_label.setObjectName("nsamples_label")

        # Days
        self.days_label = QtWidgets.QLabel(self.mode_groupbox)
        self.days_label.setGeometry(QtCore.QRect(150, 60, 91, 26))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.days_label.setFont(font)
        self.days_label.setAlignment(
            QtCore.Qt.AlignRight
            | QtCore.Qt.AlignTrailing
            | QtCore.Qt.AlignVCenter
        )
        self.days_label.setObjectName("days_label")
        self.days_label.setEnabled(False)

        self.max_days = 13
        self.days_spin = QtWidgets.QSpinBox(self.mode_groupbox)
        self.days_spin.setGeometry(QtCore.QRect(250, 61, 81, 26))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.days_spin.setFont(font)
        self.days_spin.setAlignment(
            QtCore.Qt.AlignRight
            | QtCore.Qt.AlignTrailing
            | QtCore.Qt.AlignVCenter
        )
        self.days_spin.setMaximum(self.max_days)
        self.days_spin.setMinimum(1)
        self.days_spin.setProperty("value", 1)
        self.days_spin.setObjectName("days_spin")
        self.days_spin.setEnabled(False)

        # Run button
        self.run_button = QtWidgets.QPushButton(self.hmm_groupbox)
        self.run_button.setGeometry(QtCore.QRect(340, 30, 113, 32))
        self.run_button.setObjectName("run_button")
        self.run_button.clicked.connect(self.do_viterbi)

        # Results
        self.results_groupbox = QtWidgets.QGroupBox(Dialog)
        self.results_groupbox.setGeometry(QtCore.QRect(10, 330, 460, 291))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.results_groupbox.setFont(font)
        self.results_groupbox.setObjectName("results_groupbox")

        # Samples
        self.sample_textbrowser = QtWidgets.QTextBrowser(self.results_groupbox)
        self.sample_textbrowser.setGeometry(QtCore.QRect(20, 61, 201, 171))
        self.sample_textbrowser.setObjectName("sample_textbrowser")
        self.sample_label = QtWidgets.QLabel(self.results_groupbox)
        self.sample_label.setGeometry(QtCore.QRect(20, 35, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.sample_label.setFont(font)
        self.sample_label.setObjectName("sample_label")

        # Predicted
        self.predicted_textbrowser = QtWidgets.QTextBrowser(
            self.results_groupbox
        )
        self.predicted_textbrowser.setGeometry(QtCore.QRect(240, 61, 201, 171))
        self.predicted_textbrowser.setObjectName("predicted_textbrowser")
        self.predicted_label = QtWidgets.QLabel(self.results_groupbox)
        self.predicted_label.setGeometry(QtCore.QRect(240, 35, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.predicted_label.setFont(font)
        self.predicted_label.setObjectName("predicted_label")

        # Accuracy
        self.accuracy_label = QtWidgets.QLabel(self.results_groupbox)
        self.accuracy_label.setGeometry(QtCore.QRect(20, 245, 81, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.accuracy_label.setFont(font)
        self.accuracy_label.setObjectName("accuracy_label")
        self.accuracy_value_label = QtWidgets.QLabel(self.results_groupbox)
        self.accuracy_value_label.setGeometry(QtCore.QRect(100, 245, 61, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.accuracy_value_label.setFont(font)
        self.accuracy_value_label.setAlignment(
            QtCore.Qt.AlignRight
            | QtCore.Qt.AlignTrailing
            | QtCore.Qt.AlignVCenter
        )
        self.accuracy_value_label.setObjectName("accuracy_value_label")

        self.sample_textbrowser.horizontalScrollBar().valueChanged.connect(
            self.predicted_textbrowser.horizontalScrollBar().setValue
        )
        self.sample_textbrowser.verticalScrollBar().valueChanged.connect(
            self.predicted_textbrowser.verticalScrollBar().setValue
        )
        self.predicted_textbrowser.horizontalScrollBar().valueChanged.connect(
            self.sample_textbrowser.horizontalScrollBar().setValue
        )
        self.predicted_textbrowser.verticalScrollBar().valueChanged.connect(
            self.sample_textbrowser.verticalScrollBar().setValue
        )

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Smarthouse"))
        self.processing_groupbox.setTitle(_translate("Dialog", "Preprocessing"))
        self.slice_label.setText(_translate("Dialog", "Time slice (s)"))
        self.period_checkbox.setText(_translate("Dialog", "Use day period  "))
        self.process_button.setText(_translate("Dialog", "Process"))
        self.location_checkbox.setText(_translate("Dialog", "Location only"))
        self.place_checkbox.setText(_translate("Dialog", "Place only"))
        self.hmm_groupbox.setTitle(_translate("Dialog", "Hidden Markov Model"))
        self.dataset_groupbox.setTitle(_translate("Dialog", "Dataset"))
        self.a_radio.setText(_translate("Dialog", "A"))
        self.b_radio.setText(_translate("Dialog", "B"))
        self.mode_groupbox.setTitle(_translate("Dialog", "Mode"))
        self.sampling_radio.setText(_translate("Dialog", "Random Sampling"))
        self.split_radio.setText(_translate("Dialog", "Split dataset"))
        self.nsamples_label.setText(_translate("Dialog", "# Samples"))
        self.days_label.setText(_translate("Dialog", "# Test Days"))
        self.run_button.setText(_translate("Dialog", "Run"))
        self.results_groupbox.setTitle(_translate("Dialog", "Results"))
        self.accuracy_label.setText(_translate("Dialog", "Accuracy:"))
        self.sample_label.setText(_translate("Dialog", "Sample"))
        self.predicted_label.setText(_translate("Dialog", "Predicted"))
        self.accuracy_value_label.setText(_translate("Dialog", "0.00"))
