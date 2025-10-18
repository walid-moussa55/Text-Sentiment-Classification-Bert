import tkinter as tk
from tkinter import ttk, messagebox
import threading
from model.bert_sentiment_classifier import Bert_model

# Model configuration
MODEL_PATH = './model/bert_sentiment_classifer.pth'

def predict_sentiment(input_text):
    """
    Predict sentiment for the given input text using the pre-trained BERT model.
    Args:
        input_text (str): The text to analyze.
    Returns:
        int: Predicted sentiment label (1 for positive, 0 for negative).
    """
    # Initialize model using existing class
    model = Bert_model()
    # Load pre-trained weights
    try:
        model.load_model(MODEL_PATH)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Make prediction using existing predict_sentiment method
    predicted_sentiment = model.predict_sentiment(input_text)
    return predicted_sentiment


class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, bg_color, hover_color, text_color='white', width=200, height=45, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'], highlightthickness=0, **kwargs)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text
        self.width = width
        self.height = height
        self.is_disabled = False

        self.draw_button(bg_color)
        self.bind('<Button-1>', self.on_click)
        self.bind('<Enter>', self.on_hover)
        self.bind('<Leave>', self.on_leave)

    def draw_button(self, color):
        self.delete('all')
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, radius=8, fill=color, outline='')
        self.create_text(self.width/2, self.height/2, text=self.text, fill=self.text_color,
                        font=('Segoe UI', 11, 'bold'))

    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1,
                  x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2,
                  x2-radius, y2, x2-radius, y2, x1+radius, y2, x1+radius, y2, x1, y2,
                  x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_hover(self, event):
        if not self.is_disabled:
            self.draw_button(self.hover_color)
            self.config(cursor='hand2')

    def on_leave(self, event):
        if not self.is_disabled:
            self.draw_button(self.bg_color)
            self.config(cursor='')

    def on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()

    def set_state(self, state):
        self.is_disabled = (state == 'disabled')
        if self.is_disabled:
            self.draw_button('#CCCCCC')
            self.config(cursor='')
        else:
            self.draw_button(self.bg_color)

    def set_text(self, text):
        self.text = text
        self.draw_button(self.bg_color if not self.is_disabled else '#CCCCCC')


class SentimentAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BERT Sentiment Analyzer - AI Classification")
        self.root.geometry("1000x650")
        self.root.minsize(900, 600)
        self.root.configure(bg='#F5F7FA')

        # Color scheme
        self.colors = {
            'primary': '#4A90E2',
            'primary_dark': '#357ABD',
            'success': '#5CB85C',
            'success_dark': '#4A9D4A',
            'danger': '#E74C3C',
            'bg_main': '#F5F7FA',
            'bg_card': '#FFFFFF',
            'text_primary': '#2C3E50',
            'text_secondary': '#7F8C8D',
            'border': '#E1E8ED'
        }

        self.create_widgets()
        self.center_window()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        header_label = tk.Label(header, text="📊 BERT Sentiment Analyzer",
                               font=('Segoe UI', 24, 'bold'),
                               bg=self.colors['primary'], fg='white')
        header_label.pack(pady=20)

        subtitle = tk.Label(header, text="AI-Powered Text Sentiment Classification System",
                           font=('Segoe UI', 10),
                           bg=self.colors['primary'], fg='#E8F4FD')
        subtitle.pack()

        # Main container with fixed structure
        main_container = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Single panel layout - full width
        main_panel = tk.Frame(main_container, bg=self.colors['bg_card'],
                             relief=tk.FLAT, bd=0)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Add shadow effect
        self.add_card_style(main_panel)

        # Main panel content with padding
        main_content = tk.Frame(main_panel, bg=self.colors['bg_card'])
        main_content.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # Text input section
        input_section = tk.LabelFrame(main_content, text="  Text Input  ",
                                     font=('Segoe UI', 10, 'bold'),
                                     bg=self.colors['bg_card'],
                                     fg=self.colors['text_primary'],
                                     relief=tk.FLAT)
        input_section.pack(fill=tk.X, pady=(0, 12))

        input_inner = tk.Frame(input_section, bg=self.colors['bg_card'])
        input_inner.pack(fill=tk.X, padx=12, pady=10)

        input_label = tk.Label(input_inner, text="Enter text to analyze:",
                               font=('Segoe UI', 9, 'bold'),
                               bg=self.colors['bg_card'],
                               fg=self.colors['text_primary'])
        input_label.pack(anchor='w', pady=(0, 5))

        # Text widget for multi-line input with hint
        self.text_input = tk.Text(input_inner, height=8, wrap=tk.WORD,
                                  font=('Segoe UI', 9), bg='#F8F9FA',
                                  relief=tk.FLAT, bd=0, padx=8, pady=8)
        self.text_input.pack(fill=tk.X)

        # Initialize with hint text
        self.hint_text = "Type or paste your text here..."
        self.text_input.insert(tk.END, self.hint_text)
        self.text_input.config(fg='#999999', font=('Segoe UI', 9, 'italic'))
        self.has_hint = True

        # Bind events for hint functionality
        self.text_input.bind('<FocusIn>', self.on_text_focus_in)
        self.text_input.bind('<FocusOut>', self.on_text_focus_out)
        self.text_input.bind('<Key>', self.on_text_key)

        # Clear button
        clear_frame = tk.Frame(input_inner, bg=self.colors['bg_card'])
        clear_frame.pack(fill=tk.X, pady=(8, 0))

        self.clear_btn = ModernButton(clear_frame, "🗑️  Clear Text",
                                       self.clear_text,
                                       '#95A5A6',
                                       '#7F8C8D',
                                       width=800, height=32)
        self.clear_btn.pack()

        # Predict button
        predict_frame = tk.Frame(main_content, bg=self.colors['bg_card'])
        predict_frame.pack(pady=12)

        self.predict_btn = ModernButton(predict_frame, "🔍  Analyze Sentiment",
                                       self.predict,
                                       self.colors['success'],
                                       self.colors['success_dark'],
                                       width=800, height=42)
        self.predict_btn.pack()

        # Progress bar (hidden by default)
        self.progress_frame = tk.Frame(main_content, bg=self.colors['bg_card'])

        self.progress = ttk.Progressbar(self.progress_frame,
                                       orient="horizontal",
                                       length=800,
                                       mode="indeterminate")
        self.progress.pack()

        # Result section with fixed height
        result_section = tk.LabelFrame(main_content, text="  Analysis Result  ",
                                      font=('Segoe UI', 10, 'bold'),
                                      bg=self.colors['bg_card'],
                                      fg=self.colors['text_primary'],
                                      relief=tk.FLAT,
                                      height=120)
        result_section.pack(fill=tk.X, pady=(12, 0))
        result_section.pack_propagate(False)

        result_inner = tk.Frame(result_section, bg=self.colors['bg_card'])
        result_inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self.result_label = tk.Label(result_inner,
                                     text="Enter text and click 'Analyze Sentiment' to begin",
                                     font=('Segoe UI', 10),
                                     bg=self.colors['bg_card'],
                                     fg=self.colors['text_secondary'],
                                     wraplength=800,
                                     justify=tk.CENTER)
        self.result_label.pack(expand=True)

        # Status section
        status_frame = tk.LabelFrame(main_content, text="  Status  ",
                                    font=('Segoe UI', 10, 'bold'),
                                    bg=self.colors['bg_card'],
                                    fg=self.colors['text_primary'],
                                    relief=tk.FLAT)
        status_frame.pack(fill=tk.X, pady=(12, 0))

        status_inner = tk.Frame(status_frame, bg=self.colors['bg_card'])
        status_inner.pack(fill=tk.X, padx=12, pady=10)

        self.status_label = tk.Label(status_inner,
                                    text="Ready to analyze text",
                                    font=('Segoe UI', 10),
                                    bg=self.colors['bg_card'],
                                    fg=self.colors['success'],
                                    wraplength=800,
                                    justify=tk.CENTER)
        self.status_label.pack(expand=True)

    def add_card_style(self, widget):
        """Add modern card shadow effect"""
        widget.config(highlightthickness=1, highlightbackground='#E1E8ED')

    def on_text_focus_in(self, event):
        """Hide hint text when user focuses in"""
        if self.has_hint:
            self.text_input.delete(1.0, tk.END)
            self.text_input.config(fg=self.colors['text_primary'], font=('Segoe UI', 9))
            self.has_hint = False

    def on_text_focus_out(self, event):
        """Show hint text if text box is empty"""
        if not self.text_input.get(1.0, tk.END).strip():
            self.text_input.insert(tk.END, self.hint_text)
            self.text_input.config(fg='#999999', font=('Segoe UI', 9, 'italic'))
            self.has_hint = True

    def on_text_key(self, event):
        """Handle key events for hint text"""
        if self.has_hint and event.char:
            self.text_input.delete(1.0, tk.END)
            self.text_input.config(fg=self.colors['text_primary'], font=('Segoe UI', 9))
            self.has_hint = False

    def clear_text(self):
        self.text_input.delete(1.0, tk.END)
        # Reset hint text
        self.text_input.insert(tk.END, self.hint_text)
        self.text_input.config(fg='#999999', font=('Segoe UI', 9, 'italic'))
        self.has_hint = True
        self.result_label.config(
            text="Text cleared. Enter new text to analyze.",
            fg=self.colors['text_secondary']
        )

    def predict(self):
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return

        # Update UI for prediction
        self.predict_btn.set_state('disabled')
        self.predict_btn.set_text("⏳  Analyzing...")
        self.clear_btn.set_state('disabled')

        self.progress_frame.pack(pady=10)
        self.progress.start(10)

        self.result_label.config(
            text="🔄 Processing text...\nThis may take a few moments",
            fg=self.colors['primary']
        )
        self.status_label.config(text="Analyzing sentiment...", fg=self.colors['primary'])

        # Run prediction in separate thread
        thread = threading.Thread(target=self.run_prediction, args=(text,))
        thread.daemon = True
        thread.start()

    def run_prediction(self, text):
        try:
            sentiment = predict_sentiment(text)
            self.root.after(0, self.update_result, sentiment, True)
        except Exception as e:
            self.root.after(0, self.update_result, str(e), False)

    def update_result(self, result, success):
        self.progress.stop()
        self.progress_frame.pack_forget()

        self.predict_btn.set_state('normal')
        self.predict_btn.set_text("🔍  Analyze Sentiment")
        self.clear_btn.set_state('normal')

        if success:
            sentiment = result
            if sentiment == 1:
                self.result_label.config(
                    text="😊 POSITIVE SENTIMENT\n\nThe text expresses positive emotion",
                    fg=self.colors['success'],
                    font=('Segoe UI', 12, 'bold'),
                    wraplength=300
                )
                self.status_label.config(text="Analysis complete - Positive sentiment detected", fg=self.colors['success'])
            else:
                self.result_label.config(
                    text="😔 NEGATIVE SENTIMENT\n\nThe text expresses negative emotion",
                    fg=self.colors['danger'],
                    font=('Segoe UI', 12, 'bold'),
                    wraplength=300
                )
                self.status_label.config(text="Analysis complete - Negative sentiment detected", fg=self.colors['danger'])
        else:
            self.result_label.config(
                text=f"❌ Analysis Error:\n\n{result}",
                fg=self.colors['danger'],
                font=('Segoe UI', 10),
                wraplength=300
            )
            self.status_label.config(text="Error occurred during analysis", fg=self.colors['danger'])


def main():
    root = tk.Tk()
    app = SentimentAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
