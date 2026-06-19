import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_system_architecture():
    print("Drawing System Architecture Diagram...")
    os.makedirs("reports", exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title
    ax.text(8, 10.5, "Smart Traffic Management Platform - ML Architecture Diagram", 
            fontsize=18, fontweight='bold', ha='center', va='center', color='#1e293b')
    
    # Helper function to draw a box
    def draw_box(ax, x, y, width, height, text, bg_color, border_color='#475569', text_color='#ffffff', font_size=10, is_bold=True):
        rect = patches.FancyBboxPatch(
            (x, y), width, height, 
            boxstyle="round,pad=0.1",
            edgecolor=border_color, facecolor=bg_color, 
            linewidth=1.5, mutation_scale=0.4
        )
        ax.add_patch(rect)
        weight = 'bold' if is_bold else 'normal'
        ax.text(x + width/2, y + height/2, text, 
                fontsize=font_size, fontweight=weight, 
                ha='center', va='center', color=text_color, wrap=True)
        return x, y, width, height
        
    # Helper to draw arrows
    def draw_arrow(ax, x1, y1, x2, y2, color='#475569', label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=15))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.1, label, fontsize=8, color=color, ha='center')

    # Define color scheme (harmonious HSL-like colors)
    color_input = '#475569'       # Slate gray
    color_prep = '#0284c7'        # Sky blue
    color_feat = '#0d9488'        # Teal
    color_selection = '#2563eb'   # Royal blue
    color_models = '#7c3aed'      # Purple
    color_meta = '#db2777'        # Pink/magenta
    color_engines = '#ea580c'     # Orange
    color_interface = '#16a34a'   # Green
    
    # 1. INPUT DATA
    draw_box(ax, 0.5, 8.0, 2.0, 1.0, "Astram Traffic\nIncident Dataset\n(CSV)", color_input)
    
    # 2. DATA PREPROCESSING
    draw_box(ax, 3.2, 8.0, 2.2, 1.0, "Data Preprocessing\n& Validation\n(Cleaning & Checks)", color_prep)
    draw_arrow(ax, 2.6, 8.5, 3.1, 8.5)
    
    # 3. TARGET ENGINEERING
    draw_box(ax, 6.0, 8.0, 2.2, 1.0, "Target Engineering\n(Closure, Duration,\nCongestion Score)", color_prep)
    draw_arrow(ax, 5.5, 8.5, 5.9, 8.5)
    
    # 4. FEATURE ENGINEERING
    draw_box(ax, 8.8, 8.0, 2.5, 1.0, "Feature Engineering\n- Time & Spatial\n- Corridor Graph\n- Group Aggregations", color_feat)
    draw_arrow(ax, 8.3, 8.5, 8.7, 8.5)
    
    # 5. FEATURE SELECTION
    draw_box(ax, 11.9, 8.0, 2.2, 1.0, "SHAP Feature\nSelection &\nPruning Pipeline", color_selection)
    draw_arrow(ax, 11.4, 8.5, 11.8, 8.5)
    
    # 6. ENSEMBLE BASE ESTIMATORS (Level 0)
    # Draw a bounding frame for base estimators
    frame = patches.Rectangle((0.5, 3.7), 6.5, 3.3, fill=False, edgecolor='#a78bfa', linestyle='--', linewidth=1.5)
    ax.add_patch(frame)
    ax.text(3.75, 6.75, "GroupKFold CV Base Estimators (Level 0)", fontsize=11, color='#7c3aed', fontweight='bold', ha='center')
    
    draw_box(ax, 0.8, 5.8, 1.7, 0.6, "CatBoost", color_models)
    draw_box(ax, 2.8, 5.8, 1.7, 0.6, "LightGBM", color_models)
    draw_box(ax, 4.8, 5.8, 1.7, 0.6, "XGBoost", color_models)
    draw_box(ax, 1.8, 4.8, 1.7, 0.6, "Random Forest", color_models)
    draw_box(ax, 3.8, 4.8, 1.7, 0.6, "Extra Trees", color_models)
    
    # Draw arrow from selection to base estimators
    draw_arrow(ax, 13.0, 7.9, 13.0, 7.3)
    ax.plot([13.0, 3.75], [7.3, 7.3], color='#475569', lw=1.8)
    draw_arrow(ax, 3.75, 7.3, 3.75, 7.1)
    
    # 7. STACKING META-MODELS (Level 1)
    draw_box(ax, 8.5, 6.0, 2.4, 0.7, "Primary Classifier\n(Road Closure)\nCalibrated LR Meta", color_meta)
    draw_box(ax, 11.5, 6.0, 2.4, 0.7, "Secondary Regressor\n(Clearance Duration)\nRidge Regression Meta", color_meta)
    draw_box(ax, 8.5, 4.2, 2.4, 0.7, "Secondary Regressor\n(Congestion Index)\nRidge Regression Meta", color_meta)
    draw_box(ax, 11.5, 4.2, 2.4, 0.7, "Secondary Classifier\n(Incident Severity)\nMultinomial LR Meta", color_meta)
    
    # Draw arrows from base models to meta models
    ax.plot([7.1, 7.8], [5.3, 5.3], color='#475569', lw=1.8)
    ax.plot([7.8, 7.8], [6.3, 4.5], color='#475569', lw=1.8)
    draw_arrow(ax, 7.8, 6.3, 8.4, 6.3)
    draw_arrow(ax, 7.8, 4.5, 8.4, 4.5)
    
    ax.plot([7.8, 11.2], [6.3, 6.3], color='#475569', lw=1.8)
    draw_arrow(ax, 11.2, 6.3, 11.4, 6.3)
    ax.plot([7.8, 11.2], [4.5, 4.5], color='#475569', lw=1.8)
    draw_arrow(ax, 11.2, 4.5, 11.4, 4.5)
    
    # 8. DECISION & OPTIMIZATION ENGINES
    draw_box(ax, 1.0, 1.5, 2.8, 1.0, "Emergency Priority Score\n& Resource Allocation\n(Police, Tow Trucks, Ambulances)", color_engines)
    draw_box(ax, 4.5, 1.5, 2.8, 1.0, "Dijkstra Diversion Router\n(Route Score: Congestion +\nCentrality + Frequency)", color_engines)
    draw_box(ax, 8.0, 1.5, 2.8, 1.0, "Incident Similarity Retriever\n(Cosine Distance + KNN\non PCA Transformed Space)", color_engines)
    draw_box(ax, 11.5, 1.5, 2.8, 1.0, "Explainable AI (XAI)\n(Narrative Compiler &\nSHAP Feature Drivers)", color_engines)
    
    # Draw arrows from meta models to engines
    # Primary/Secondary outputs flow down to decision engines
    ax.plot([9.7, 9.7], [3.9, 3.2], color='#475569', lw=1.8)
    ax.plot([12.7, 12.7], [3.9, 3.2], color='#475569', lw=1.8)
    ax.plot([9.7, 12.7], [3.2, 3.2], color='#475569', lw=1.8)
    ax.plot([11.2, 11.2], [3.2, 2.9], color='#475569', lw=1.8)
    ax.plot([2.4, 12.9], [2.9, 2.9], color='#475569', lw=1.8)
    draw_arrow(ax, 2.4, 2.9, 2.4, 2.6)
    draw_arrow(ax, 5.9, 2.9, 5.9, 2.6)
    draw_arrow(ax, 9.4, 2.9, 9.4, 2.6)
    draw_arrow(ax, 12.9, 2.9, 12.9, 2.6)
    
    # 9. OUTPUT / INTERFACES
    draw_box(ax, 4.0, 0.1, 3.5, 0.7, "Platform REST APIs\n(/api/predict, /api/metrics)", color_interface)
    draw_box(ax, 8.5, 0.1, 3.5, 0.7, "Interactive Portal Dashboard\n(Web UI Visualization)", color_interface)
    
    # Draw arrows from engines to output
    ax.plot([2.4, 2.4], [1.4, 1.0], color='#475569', lw=1.8)
    ax.plot([5.9, 5.9], [1.4, 1.0], color='#475569', lw=1.8)
    ax.plot([9.4, 9.4], [1.4, 1.0], color='#475569', lw=1.8)
    ax.plot([12.9, 12.9], [1.4, 1.0], color='#475569', lw=1.8)
    ax.plot([2.4, 12.9], [1.0, 1.0], color='#475569', lw=1.8)
    draw_arrow(ax, 5.75, 1.0, 5.75, 0.9)
    draw_arrow(ax, 10.25, 1.0, 10.25, 0.9)
    
    # Save figure
    plt.savefig("reports/system_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved reports/system_architecture.png successfully!")

if __name__ == "__main__":
    draw_system_architecture()
