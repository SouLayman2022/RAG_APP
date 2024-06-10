import sys
import fitz  # PyMuPDF
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QLineEdit, QScrollArea, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QPainterPath, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from llama_cpp import Llama

class ImageLabel(QLabel):
    def __init__(self, image_path, shape="circle"):
        super().__init__()
        self.image_path = image_path
        self.shape = shape
        self.setFixedSize(120, 120)  # Set fixed size for student images
        if shape == "rectangle":
            self.setFixedSize(240, 178)  # Set fixed size for school logo

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.shape == "circle":
            path = QPainterPath()
            path.addEllipse(0, 0, self.width(), self.height())
            painter.setClipPath(path)

        pixmap = QPixmap(self.image_path).scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        painter.drawPixmap(self.rect(), pixmap)

class PDFQAApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Intelligence Artificielle Générative")
        self.setGeometry(100, 100, 600, 400)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                border: none;
            }
            QPushButton {
                background-color: #2c3e50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QLineEdit {
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QTextEdit {
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
        """)

        # Main layout
        main_layout = QVBoxLayout()

        # Top layout for pictures and logo
        top_layout = QHBoxLayout()

        # My picture
        student_image_path = "./Assets/SoSo.png"
        student_label = ImageLabel(student_image_path, shape="circle")
        top_layout.addWidget(student_label)

        top_layout.addStretch()

        # School logo
        logo_label = ImageLabel("./Assets/llama3.jpg", shape="rectangle")
        top_layout.addWidget(logo_label, alignment=Qt.AlignRight)

        main_layout.addLayout(top_layout)

        # Bottom layout for question input and import button
        bottom_layout = QHBoxLayout()

        # Text field for questions
        self.question_edit = QLineEdit(self)
        self.question_edit.setPlaceholderText("Enter your question here...")
        bottom_layout.addWidget(self.question_edit)

        # Ask button
        self.ask_button = QPushButton("Ask Question", self)
        self.ask_button.clicked.connect(self.ask_question)
        bottom_layout.addWidget(self.ask_button)

        bottom_layout.addStretch()

        # Import button
        self.import_button = QPushButton("Import PDF", self)
        self.import_button.clicked.connect(self.import_pdf)
        bottom_layout.addWidget(self.import_button)

        bottom_layout.addStretch()

        # Load model button
        self.load_model_button = QPushButton("Load Model", self)
        self.load_model_button.clicked.connect(self.load_model)
        bottom_layout.addWidget(self.load_model_button)

        main_layout.addLayout(bottom_layout)

        # Scrollable text edit for displaying the answer
        scroll_area = QScrollArea()
        self.answer_edit = QTextEdit(self)
        self.answer_edit.setReadOnly(True)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.answer_edit)
        scroll_area.setFixedHeight(150)  # Make the answer section small and scrollable
        main_layout.addWidget(scroll_area)

        # Links layout
        links_layout = QHBoxLayout()

        github_label = QLabel('<a href="https://github.com/SouLayman2022">GitHub</a>', self)
        github_label.setOpenExternalLinks(True)
        links_layout.addWidget(github_label)

        linkedin_label = QLabel('<a href="https://www.linkedin.com/in/soulayman-el-guasmi-13b890240/">LinkedIn</a>', self)
        linkedin_label.setOpenExternalLinks(True)
        links_layout.addWidget(linkedin_label)

        main_layout.addLayout(links_layout)

        # Set main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.model = None

    def load_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select LLaMA Model File", "", "All Files (*)", options=options)
        if file_name:
            self.model = Llama(model_path=file_name)
            QMessageBox.information(self, "Model Loaded", "LLaMA model loaded successfully!")

    def import_pdf(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PDF File", "", "PDF Files (*.pdf);;All Files (*)", options=options)
        if file_name:
            self.extract_text_from_pdf(file_name)

    def extract_text_from_pdf(self, file_path):
        pdf_document = fitz.open(file_path)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        self.pdf_text = text

    def split_text_into_chunks(self, text, max_tokens):
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i + max_tokens])
            chunks.append(chunk)
        return chunks

    def ask_question(self):
        if self.model is None:
            self.load_model()

        if self.model is not None:
            question = self.question_edit.text()
            if question and hasattr(self, 'pdf_text'):
                context_window = 512
                question_tokens = len(question.split())
                max_context_tokens = context_window - question_tokens - 10  # Leave some buffer

                chunks = self.split_text_into_chunks(self.pdf_text, max_context_tokens)

                responses = []
                for chunk in chunks:
                    prompt = f"{chunk}\n\nQuestion: {question}\nAnswer:"
                    # Interact with the LLaMA model
                    try:
                        response = self.model.create_completion(prompt, max_tokens=context_window)
                        responses.append(response['choices'][0]['text'])
                    except ValueError as e:
                        print(f"Skipping chunk due to error: {e}")

                combined_response = "\n".join(responses)
                self.answer_edit.setPlainText(combined_response)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PDFQAApp()
    window.show()
    sys.exit(app.exec_())
