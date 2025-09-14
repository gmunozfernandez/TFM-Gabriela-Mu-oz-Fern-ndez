"""
Movie Success Prediction Application - Improved GUI
Interfaz gráfica mejorada para predecir el éxito comercial de películas
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import threading
import time
import math
import sys
import os
from pathlib import Path
import logging
import random

np.random.seed(42)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# FUNCIONES AUXILIARES MOCK PARA PRUEBAS
# ==========================================

def mock_predict_movie_success(movie_data):
    """Función mock para simular predicción sin cargar modelos reales"""
    time.sleep(1)
    budget = movie_data.get('budget', 50000000)
    runtime = movie_data.get('runtime', 120)
    year = movie_data.get('year', 2024)
    base_prob = 0.5
    if budget > 100000000: base_prob += 0.2
    elif budget > 50000000: base_prob += 0.1
    elif budget < 10000000: base_prob -= 0.2
    if 90 <= runtime <= 150: base_prob += 0.1
    elif runtime > 180: base_prob -= 0.1
    if year >= 2020: base_prob += 0.05

    probability = max(0.05, min(0.95, base_prob))
    confidence = abs(probability - 0.5)
    confidence = min(0.8, confidence)

    return probability, confidence


# ==========================================
# CLASE PRINCIPAL
# ==========================================

class MoviePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Movie Success Predictor")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a2e')

        # Variables animación
        self.animation_running = False
        self.probability_value = 0.0
        self.confidence_value = 0.0
        self.is_predicting = False

        # Estilos
        self.setup_style()

        # UI
        self.setup_ui()
        self.draw_initial_canvas()
        self.status_var.set("✅ Ready to predict movie success!")

    # ---------- Estilos ----------
    def setup_style(self):
        style = ttk.Style()
        try: style.theme_use('clam')
        except: pass
        style.configure('Card.TFrame', background='#0f0f23', relief='flat')
        style.configure('Card.TLabel', background='#0f0f23', foreground='#eaeaea')
        style.configure('Title.TLabel', background='#0f0f23', foreground='#ffd700', font=("Arial", 16, "bold"))
        style.configure('KPI.TLabel', background='#1b1b35', foreground='#cccccc', padding=6)
        style.configure('PB.Success.Horizontal.TProgressbar', troughcolor='#1b1b35', background='#44ff44')
        style.configure('PB.Warn.Horizontal.TProgressbar', troughcolor='#1b1b35', background='#ffaa00')
        style.configure('PB.Danger.Horizontal.TProgressbar', troughcolor='#1b1b35', background='#ff4444')

    def pick_color_style(self, v):
        if v < 0.3: return 'PB.Danger.Horizontal.TProgressbar'
        elif v < 0.7: return 'PB.Warn.Horizontal.TProgressbar'
        return 'PB.Success.Horizontal.TProgressbar'

    def get_category(self, p, c):
        if c < 0.15: return ("UNCERTAIN", "#6b6b6b", "🤔")
        if p > 0.7: return ("HIGH SUCCESS", "#1f8b24", "🎯")
        if p > 0.5: return ("MODERATE SUCCESS", "#2c7be5", "👍")
        if p > 0.3: return ("MIXED SIGNALS", "#c68a00", "⚖️")
        return ("HIGH RISK", "#a83232", "⚠️")

    # ---------- UI ----------
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Título
        title_label = tk.Label(main_frame, text="🎬 MOVIE SUCCESS PREDICTOR",
                               font=("Arial", 28, "bold"), bg='#1a1a2e', fg='#ffd700')
        title_label.pack()
        subtitle_label = tk.Label(main_frame,
                                  text="Predict the commercial success of your movie project",
                                  font=("Arial", 12), bg='#1a1a2e', fg='#cccccc')
        subtitle_label.pack(pady=(5, 20))

        # Entrada
        self.create_input_section(main_frame)

        # Resultados
        self.create_results_section(main_frame)

        # Status bar
        self.create_status_bar()

    def create_input_section(self, parent):
        input_frame = tk.LabelFrame(parent, text="📝 Movie Information",
                                    font=("Arial", 12, "bold"),
                                    bg='#16213e', fg='#ffd700')
        input_frame.pack(fill=tk.X, pady=(0, 20))
        content = tk.Frame(input_frame, bg='#16213e'); content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        left = tk.Frame(content, bg='#16213e'); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,20))
        right = tk.Frame(content, bg='#16213e'); right.pack(side=tk.RIGHT, fill=tk.Y)
        self.create_text_fields(left)
        self.create_numeric_fields(right)
        self.create_action_buttons(right)

    def create_text_fields(self, parent):
        tk.Label(parent, text="🎬 Movie Title", font=("Arial", 11, "bold"),
                 bg='#16213e', fg='#ffd700').pack(anchor=tk.W)
        self.title_var = tk.StringVar(value="The Amazing Adventure")
        tk.Entry(parent, textvariable=self.title_var, font=("Arial",10),
                 bg='#2a2a3e', fg='#fff').pack(fill=tk.X, pady=5, ipady=5)

        tk.Label(parent, text="🎭 Genres", font=("Arial", 11, "bold"),
                 bg='#16213e', fg='#ffd700').pack(anchor=tk.W, pady=(10,0))
        self.genres_var = tk.StringVar(value="Action, Adventure, Sci-Fi")
        tk.Entry(parent, textvariable=self.genres_var, font=("Arial",10),
                 bg='#2a2a3e', fg='#fff').pack(fill=tk.X, pady=5, ipady=5)

        tk.Label(parent, text="📝 Description", font=("Arial", 11, "bold"),
                 bg='#16213e', fg='#ffd700').pack(anchor=tk.W, pady=(10,0))
        self.text_content = scrolledtext.ScrolledText(parent, height=6, font=("Arial",10),
                                                     bg='#2a2a3e', fg='#fff')
        self.text_content.pack(fill=tk.BOTH, expand=True)
        self.text_content.insert("1.0", "Sci-fi adventure with a mysterious portal...")

    def create_numeric_fields(self, parent):
        tk.Label(parent, text="💰 Budget & Details", font=("Arial", 11, "bold"),
                 bg='#16213e', fg='#ffd700').pack(anchor=tk.W)
        fields = [("💵 Budget (USD):", "budget_var", "75000000"),
                  ("⏱️ Runtime (minutes):", "runtime_var", "125"),
                  ("📅 Release Year:", "year_var", "2024"),
                  ("📆 Release Month:", "month_var", "7")]
        for lbl, varname, default in fields:
            tk.Label(parent, text=lbl, bg='#16213e', fg='#fff').pack(anchor=tk.W, pady=(10,0))
            v = tk.StringVar(value=default); setattr(self, varname, v)
            tk.Entry(parent, textvariable=v, bg='#2a2a3e', fg='#fff').pack(fill=tk.X)

    def create_action_buttons(self, parent):
        btn = tk.Button(parent, text="🔮 PREDICT SUCCESS", command=self.predict_async,
                        font=("Arial",14,"bold"), bg='#ffd700', fg='#1a1a2e')
        btn.pack(fill=tk.X, pady=(20,5)); self.predict_btn = btn
        tk.Button(parent, text="📄 Load Example", command=self.load_example,
                  bg='#4a4a6e', fg='#fff').pack(fill=tk.X, pady=5)
        tk.Button(parent, text="🗑️ Clear All", command=self.clear_fields,
                  bg='#6e4a4a', fg='#fff').pack(fill=tk.X, pady=5)

    def create_results_section(self, parent):
        results = tk.LabelFrame(parent, text="📊 Prediction Results",
                                font=("Arial", 12, "bold"),
                                bg='#16213e', fg='#ffd700')
        results.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        container = tk.Frame(results, bg='#16213e'); container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        container.grid_columnconfigure(0, weight=1); container.grid_columnconfigure(1, weight=1)

        # Gauge
        left = tk.Frame(container, bg='#16213e'); left.grid(row=0, column=0, sticky="nsew", padx=(0,12))
        self.canvas = tk.Canvas(left, width=520, height=320, bg='#0f0f23', highlightthickness=2)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Tarjeta lateral
        right = ttk.Frame(container, style='Card.TFrame', padding=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(12,0))
        self.category_var = tk.StringVar(value="—")
        self.category_chip = tk.Label(right, textvariable=self.category_var,
                                      bg='#1b1b35', fg='#fff', font=("Arial",10,"bold"))
        self.category_chip.pack(anchor="w")
        self.big_prob = ttk.Label(right, text="—", style='Title.TLabel'); self.big_prob.pack(anchor="w", pady=10)
        ttk.Label(right, text="Success Probability", style='Card.TLabel').pack(anchor="w")
        self.prob_bar = ttk.Progressbar(right, orient='horizontal', length=280, mode='determinate',
                                        style='PB.Warn.Horizontal.TProgressbar', maximum=100)
        self.prob_bar.pack(fill='x', pady=(4,12))
        ttk.Label(right, text="Confidence", style='Card.TLabel').pack(anchor="w")
        self.conf_bar = ttk.Progressbar(right, orient='horizontal', length=280, mode='determinate',
                                        style='PB.Warn.Horizontal.TProgressbar', maximum=100)
        self.conf_bar.pack(fill='x', pady=(4,12))
        kpi_wrap = tk.Frame(right, bg='#0f0f23'); kpi_wrap.pack(fill='x', pady=(8,2))
        self.kpi_budget = ttk.Label(kpi_wrap, text="Budget: —", style='KPI.TLabel')
        self.kpi_runtime = ttk.Label(kpi_wrap, text="Runtime: —", style='KPI.TLabel')
        self.kpi_year = ttk.Label(kpi_wrap, text="Year: —", style='KPI.TLabel')
        self.kpi_budget.pack(side='left', padx=5); self.kpi_runtime.pack(side='left', padx=5); self.kpi_year.pack(side='left', padx=5)

        labels = tk.Frame(right, bg='#0f0f23'); labels.pack(fill='x', pady=(12,0))
        self.probability_label = tk.Label(labels, text="Success Probability: --", font=("Arial",12,"bold"),
                                          bg='#0f0f23', fg='#ffd700'); self.probability_label.pack(anchor=tk.W)
        self.confidence_label = tk.Label(labels, text="Confidence Level: --", font=("Arial",12,"bold"),
                                         bg='#0f0f23', fg='#ffd700'); self.confidence_label.pack(anchor=tk.W)

        rec_frame = tk.Frame(results, bg='#16213e'); rec_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        tk.Label(rec_frame, text="💡 Recommendation", font=("Arial",12,"bold"), bg='#16213e', fg='#ffd700').pack(anchor=tk.W)
        self.recommendation_text = scrolledtext.ScrolledText(rec_frame, height=6, font=("Arial",11),
                                                            bg='#2a2a3e', fg='#fff'); self.recommendation_text.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 relief=tk.SUNKEN, anchor=tk.W,
                 bg='#16213e', fg='#fff', font=("Arial",9)).pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- Canvas ----------
    def draw_initial_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_text(260, 28, text="🎯 SUCCESS PROBABILITY ANALYZER",
                                fill='#ffd700', font=("Arial",16,"bold"))
        # self.canvas.create_text(260, 160, text="Enter movie details and click 'PREDICT SUCCESS'",
        #                         fill='#cccccc', font=("Arial",12))
        self.draw_probability_meter(0.0, 0.0)
        self.draw_scale_labels()

    def draw_scale_labels(self):
        self.canvas.create_text(100, 280, text="❌ HIGH RISK", fill='#ff4444')
        self.canvas.create_text(260, 300, text="⚠️ UNCERTAIN", fill='#ffaa00')
        self.canvas.create_text(420, 280, text="✅ SUCCESS", fill='#44ff44')

    def draw_probability_meter(self, probability, confidence):
        self.canvas.delete("meter")
        cx, cy, radius = 260, 180, 100
        self.canvas.create_arc(cx-radius, cy-radius, cx+radius, cy+radius,
                               start=0, extent=180, outline='#333', width=25,
                               style=tk.ARC, tags="meter")
        if probability>0:
            extent = probability*180
            color = self.get_probability_color(probability)
            self.canvas.create_arc(cx-radius, cy-radius, cx+radius, cy+radius,
                                   start=180, extent=extent, outline=color, width=20,
                                   style=tk.ARC, tags="meter")
        angle = math.pi - (probability*math.pi)
        x = cx + (radius-20)*math.cos(angle); y = cy - (radius-20)*math.sin(angle)
        self.canvas.create_line(cx,cy,x,y,fill='#fff',width=4,tags="meter")
        self.canvas.create_text(cx, cy+50, text=f"{probability:.1%}", fill='#fff', font=("Arial",16,"bold"), tags="meter")

    def get_probability_color(self, p):
        if p < 0.3: return '#ff4444'
        elif p < 0.7: return '#ffaa00'
        return '#44ff44'

    def animate_to_result(self, target_p, target_c):
        if self.animation_running: return
        self.animation_running = True
        def animate():
            steps=40
            for i in range(steps+1):
                t=i/steps; ease=1-(1-t)**3
                p=self.probability_value+(target_p-self.probability_value)*ease
                c=self.confidence_value+(target_c-self.confidence_value)*ease
                self.root.after(0, lambda pp=p,cc=c: self.draw_probability_meter(pp,cc))
                time.sleep(0.03)
            self.probability_value=target_p; self.confidence_value=target_c
            self.animation_running=False
        threading.Thread(target=animate,daemon=True).start()

    # ---------- Predicción ----------
    def predict_async(self):
        if self.is_predicting: return
        self.is_predicting=True; self.status_var.set("🔮 Analyzing...")
        self.predict_btn.config(state='disabled', text="🔄 Analyzing...")
        def predict():
            try:
                data={'title':self.title_var.get(),'genres':self.genres_var.get(),
                      'text':self.text_content.get("1.0",tk.END).strip(),
                      'budget':float(self.budget_var.get()),'runtime':int(self.runtime_var.get()),
                      'year':int(self.year_var.get()),'month':int(self.month_var.get())}
                prob,conf=mock_predict_movie_success(data)
                rec=self.generate_recommendation(prob,conf,data)
                self.root.after(0, lambda: self.update_results(prob,conf,rec))
            except Exception as e:
                self.root.after(0, lambda: self.handle_prediction_error(str(e)))
            finally:
                self.root.after(0, self.reset_predict_button)
        threading.Thread(target=predict,daemon=True).start()

    def generate_recommendation(self, p,c,data):
        cat,_,emoji=self.get_category(p,c)
        return f"{emoji} {cat}\nProbability: {p:.1%}\nConfidence: {c:.2f}"

    def update_results(self,p,c,rec):
        self.animate_to_result(p,c)
        self.big_prob.config(text=f"{p:.1%}")
        self.probability_label.config(text=f"Success Probability: {p:.1%}", fg=self.get_probability_color(p))
        conf_color='#44ff44' if c>0.3 else ('#ffaa00' if c>0.15 else '#ff4444')
        self.confidence_label.config(text=f"Confidence Level: {c:.2f}", fg=conf_color)
        self.prob_bar['value']=p*100; self.conf_bar['value']=c*100
        self.prob_bar.configure(style=self.pick_color_style(p)); self.conf_bar.configure(style=self.pick_color_style(c))
        cat,color,emoji=self.get_category(p,c)
        self.category_var.set(f"{emoji} {cat}"); self.category_chip.config(bg=color, fg='#fff')
        self.kpi_budget.config(text=f"Budget: ${float(self.budget_var.get()):,.0f}")
        self.kpi_runtime.config(text=f"Runtime: {int(self.runtime_var.get())} min")
        self.kpi_year.config(text=f"Year: {self.year_var.get()}")
        self.recommendation_text.delete(1.0,tk.END); self.recommendation_text.insert(1.0,rec)
        self.status_var.set("✅ Prediction completed"); self.is_predicting=False

    def handle_prediction_error(self,msg):
        messagebox.showerror("Error",msg); self.status_var.set("❌ Prediction failed"); self.is_predicting=False

    def reset_predict_button(self):
        self.predict_btn.config(state='normal', text="🔮 PREDICT SUCCESS")

    def load_example(self):
        ex=[{"title":"Quantum Horizons","genres":"Sci-Fi","text":"Parallel universes...", "budget":"95000000","runtime":"142","year":"2024","month":"8"}]
        e=random.choice(ex)
        self.title_var.set(e['title']); self.genres_var.set(e['genres'])
        self.text_content.delete("1.0",tk.END); self.text_content.insert("1.0",e['text'])
        self.budget_var.set(e['budget']); self.runtime_var.set(e['runtime'])
        self.year_var.set(e['year']); self.month_var.set(e['month'])
        self.status_var.set("✅ Example loaded")

    def clear_fields(self):
        self.title_var.set(""); self.genres_var.set(""); self.text_content.delete("1.0",tk.END)
        self.budget_var.set("50000000"); self.runtime_var.set("120"); self.year_var.set("2024"); self.month_var.set("6")
        self.probability_value=0.0; self.confidence_value=0.0; self.draw_initial_canvas()
        self.probability_label.config(text="Success Probability: --", fg='#ffd700')
        self.confidence_label.config(text="Confidence Level: --", fg='#ffd700')
        self.recommendation_text.delete(1.0,tk.END)
        self.status_var.set("✅ Cleared")

# ---------- MAIN ----------
def main():
    root = tk.Tk(); app = MoviePredictorApp(root)
    root.mainloop()

if __name__=="__main__":
    main()
