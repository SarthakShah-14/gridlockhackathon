import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def generate_incident_pdf_report(event: dict, output_path: str = "reports/operational_incident_report.pdf"):
    """
    Module 14: Exportable Operational Incident Report
    Compiles predictions, resource requirements, routing suggestions, SHAP explanations,
    and system metadata into a styled, police-style PDF report using Matplotlib.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Setup clean, professional styling
    fig = plt.figure(figsize=(8.5, 11))
    
    # Remove all axes
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Draw dark header band
    rect = plt.Rectangle((0, 0.90), 1.0, 0.10, facecolor='#1e293b', alpha=1.0)
    ax.add_patch(rect)
    
    # Title Text
    fig.text(0.05, 0.94, "BENGALURU TRAFFIC MANAGEMENT CENTER", color='white', fontsize=16, fontweight='bold')
    fig.text(0.05, 0.91, "OFFICIAL OPERATIONAL INCIDENT RESPONSE REPORT", color='#38bdf8', fontsize=10, fontweight='semibold')
    
    # Metadata Block
    fig.text(0.05, 0.86, f"Incident ID: {event.get('id', 'N/A')}", fontsize=10, fontweight='bold', color='#0f172a')
    fig.text(0.05, 0.84, f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9, color='#475569')
    fig.text(0.05, 0.82, f"Model Version: {event.get('model_version', '1.0.0')}", fontsize=9, color='#475569')
    fig.text(0.50, 0.84, f"Coordinates: {event.get('latitude', 0.0):.5f}, {event.get('longitude', 0.0):.5f}", fontsize=9, color='#475569')
    fig.text(0.50, 0.82, f"Corridor / Junction: {event.get('corridor', 'N/A')} / {event.get('junction', 'N/A')}", fontsize=9, color='#475569')
    
    # Draw horizontal divider
    line = plt.Line2D([0.05, 0.95], [0.80, 0.80], color='#cbd5e1', linewidth=1.5)
    ax.add_line(line)
    
    # 2. Section: Predictive Metrics
    fig.text(0.05, 0.76, "1. INTELLECTUAL FORECAST METRICS", fontsize=12, fontweight='bold', color='#1e293b')
    
    forecast_data = [
        ["Road Closure Probability", f"{event.get('road_closure_prob', 0.0)*100:.2f}%", f"Confidence: {event.get('confidence_score', 0.0):.1f}%"],
        ["Predicted Incident Duration", f"{event.get('predicted_duration', 0.0):.1f} mins", f"Range: {event.get('duration_min_bound', 0.0):.1f} - {event.get('duration_max_bound', 0.0):.1f} mins"],
        ["Severity Level", f"{event.get('predicted_severity', 'N/A')}", "Derived from duration forecast"],
        ["Traffic Congestion Index", f"{event.get('congestion_score', 0.0):.1f} / 100", f"Category: {event.get('risk_level', 'N/A')}"]
    ]
    
    table_y = 0.62
    for idx, row in enumerate(forecast_data):
        y_pos = table_y + (3 - idx) * 0.035
        # Background block for alternating rows
        bg_color = '#f8fafc' if idx % 2 == 0 else '#ffffff'
        rect = plt.Rectangle((0.05, y_pos - 0.01), 0.90, 0.03, facecolor=bg_color, alpha=1.0)
        ax.add_patch(rect)
        
        fig.text(0.07, y_pos, row[0], fontsize=10, fontweight='semibold', color='#334155')
        fig.text(0.40, y_pos, row[1], fontsize=10, fontweight='bold', color='#0f172a')
        fig.text(0.70, y_pos, row[2], fontsize=9, color='#64748b')
        
    # Draw horizontal divider
    line = plt.Line2D([0.05, 0.95], [0.60, 0.60], color='#cbd5e1', linewidth=1.5)
    ax.add_line(line)
    
    # 3. Section: Resource Allocation Suggestions
    fig.text(0.05, 0.56, "2. OPTIMIZED MANPOWER & RESOURCE DEPLOYMENT", fontsize=12, fontweight='bold', color='#1e293b')
    
    resources_data = [
        ["Traffic Officers", f"{event.get('officers_required', 2)} units", "Min Staffing: 2 | Max Staffing: 10"],
        ["Barricades Needed", f"{event.get('barricades_required', 0)} units", "Deployed at key entry junctions"],
        ["Traffic Cones", f"{event.get('traffic_cones_required', 5)} units", "For lane narrowing and warnings"],
        ["Patrol Vehicles", f"{event.get('patrol_vehicles_required', 1)} units", "For active mobile surveillance"],
        ["Diversion Teams", f"{event.get('diversion_teams_required', 0)} units", "Deployed at alternative junctions"]
    ]
    
    table_y2 = 0.38
    for idx, row in enumerate(resources_data):
        y_pos = table_y2 + (4 - idx) * 0.035
        bg_color = '#f0f9ff' if idx % 2 == 0 else '#ffffff'
        rect = plt.Rectangle((0.05, y_pos - 0.01), 0.90, 0.03, facecolor=bg_color, alpha=1.0)
        ax.add_patch(rect)
        
        fig.text(0.07, y_pos, row[0], fontsize=10, fontweight='semibold', color='#0369a1')
        fig.text(0.40, y_pos, row[1], fontsize=10, fontweight='bold', color='#0f172a')
        fig.text(0.70, y_pos, row[2], fontsize=9, color='#64748b')
        
    # Draw horizontal divider
    line = plt.Line2D([0.05, 0.95], [0.35, 0.35], color='#cbd5e1', linewidth=1.5)
    ax.add_line(line)
    
    # 4. Section: Diversion Recommendation
    fig.text(0.05, 0.31, "3. ACTIVE ROUTE DIVERSION RECOMMENDATION", fontsize=12, fontweight='bold', color='#1e293b')
    fig.text(0.07, 0.28, f"Alternative Junction: {event.get('alternative_junction', 'None')}", fontsize=10, fontweight='bold', color='#0f172a')
    fig.text(0.07, 0.26, f"Alternative Corridor: {event.get('alternative_corridor', 'None')}", fontsize=9, color='#475569')
    fig.text(0.07, 0.24, f"Expected Congestion Reduction: {event.get('congestion_reduction', 0.0):.1f}%", fontsize=9, fontweight='semibold', color='#16a34a')
    
    # 5. Section: Narrative Summary
    fig.text(0.05, 0.20, "4. NARRATIVE DECISION EXPLANATION", fontsize=12, fontweight='bold', color='#1e293b')
    narrative = event.get('explanation_narrative', 'Typical baseline incident response.')
    
    # Draw simple wrapped text
    words = narrative.split()
    lines = []
    current_line = []
    for word in words:
        if len(" ".join(current_line + [word])) > 80:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))
        
    y_text = 0.17
    for line_text in lines[:4]:
        fig.text(0.07, y_text, line_text, fontsize=9.5, color='#334155', style='italic')
        y_text -= 0.02
        
    # Footer
    fig.text(0.05, 0.05, "CONFIDENTIAL - TMC INTERNAL DISTRIBUTION ONLY", color='#94a3b8', fontsize=8, fontweight='bold')
    fig.text(0.70, 0.05, "BENGALURU TRAFFIC PLATFORM v1.2", color='#94a3b8', fontsize=8)
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
