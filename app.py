from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Pour générer des graphiques sans interface
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (chi2_contingency, spearmanr, pearsonr, mannwhitneyu, 
                         kruskal, ttest_ind, f_oneway, shapiro, levene, 
                         normaltest, kendalltau, fisher_exact)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import os
import json
from datetime import datetime
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Créer les dossiers si nécessaire
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnalyseurStatistiqueWeb:
    def __init__(self):
        self.variables = {}
        self.data = None
        self.resultats = []
        
    def charger_donnees(self, filepath):
        """Charge le fichier de données"""
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(filepath)
            elif filepath.endswith('.txt'):
                self.data = pd.read_csv(filepath, sep='\t')
            else:
                self.data = pd.read_csv(filepath)
            
            return True, list(self.data.columns)
        except Exception as e:
            return False, str(e)
    
    def definir_variables(self, variables_config):
        """Définit les types de variables"""
        self.variables = variables_config
    
    def verifier_normalite(self, data):
        """Vérifie la normalité des données"""
        if len(data) < 3:
            return False, None
        if len(data) < 5000:
            stat, p = shapiro(data)
        else:
            stat, p = normaltest(data)
        return p > 0.05, p
    
    def verifier_homogeneite_variance(self, *groupes):
        """Vérifie l'homogénéité des variances"""
        stat, p = levene(*groupes)
        return p > 0.05, p
    
    def determiner_test(self, var1, var2, type1, type2):
        """Détermine le test statistique approprié"""
        if type1 in ['nominale', 'ordinale'] and type2 in ['nominale', 'ordinale']:
            if self.data[var1].nunique() == 2 and self.data[var2].nunique() == 2:
                contingence = pd.crosstab(self.data[var1], self.data[var2])
                if contingence.sum().sum() < 1000 and contingence.min().min() < 5:
                    return 'fisher', 'Test exact de Fisher'
            
            if type1 == 'ordinale' and type2 == 'ordinale':
                return 'spearman_ordinal', 'Corrélation de Spearman (+ Chi-2)'
            
            return 'chi2', 'Test du Chi-2'
        
        elif (type1 in ['nominale', 'ordinale'] and type2 in ['continue', 'discrète']) or \
             (type2 in ['nominale', 'ordinale'] and type1 in ['continue', 'discrète']):
            var_cat = var1 if type1 in ['nominale', 'ordinale'] else var2
            var_num = var2 if var_cat == var1 else var1
            n_groupes = self.data[var_cat].nunique()
            
            if n_groupes == 2:
                groupes = [self.data[self.data[var_cat] == cat][var_num].dropna() 
                          for cat in self.data[var_cat].unique()]
                
                normalite = all([self.verifier_normalite(g)[0] for g in groupes if len(g) > 2])
                homogeneite, _ = self.verifier_homogeneite_variance(*groupes) if len(groupes) == 2 else (True, 1)
                
                if normalite and homogeneite:
                    return 'ttest', 'Test t de Student (paramétrique)'
                else:
                    return 'mannwhitney', 'Test de Mann-Whitney U (non-paramétrique)'
            else:
                groupes = [self.data[self.data[var_cat] == cat][var_num].dropna() 
                          for cat in self.data[var_cat].unique()]
                
                normalite = all([self.verifier_normalite(g)[0] for g in groupes if len(g) > 2])
                homogeneite, _ = self.verifier_homogeneite_variance(*groupes)
                
                if normalite and homogeneite:
                    return 'anova', 'ANOVA (paramétrique)'
                else:
                    return 'kruskal', 'Test de Kruskal-Wallis (non-paramétrique)'
        
        elif type1 in ['continue', 'discrète'] and type2 in ['continue', 'discrète']:
            data_clean = self.data[[var1, var2]].dropna()
            
            normal1, _ = self.verifier_normalite(data_clean[var1])
            normal2, _ = self.verifier_normalite(data_clean[var2])
            
            if normal1 and normal2:
                return 'pearson', 'Corrélation de Pearson (paramétrique)'
            else:
                return 'spearman_kendall', 'Corrélations de Spearman et Kendall (non-paramétriques)'
        
        return None, None
    
    def plot_to_base64(self):
        """Convertit un graphique matplotlib en base64"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    
    def executer_chi2(self, var1, var2):
        """Exécute le test du Chi-2"""
        contingence = pd.crosstab(self.data[var1], self.data[var2])
        chi2, p_value, dof, expected = chi2_contingency(contingence)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingence, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Effectifs'})
        plt.title(f'Table de contingence: {var1} vs {var2}')
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Chi-2',
            'statistique': float(chi2),
            'p_value': float(p_value),
            'ddl': int(dof),
            'effectif_theorique_min': float(expected.min()),
            'graphique': graphique
        }
    
    def executer_fisher(self, var1, var2):
        """Exécute le test exact de Fisher"""
        contingence = pd.crosstab(self.data[var1], self.data[var2])
        oddsratio, p_value = fisher_exact(contingence)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingence, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Effectifs'})
        plt.title(f'Table de contingence: {var1} vs {var2}\n(Test exact de Fisher)')
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Test exact de Fisher',
            'statistique': float(oddsratio),
            'p_value': float(p_value),
            'graphique': graphique
        }
    
    def executer_spearman_ordinal(self, var1, var2):
        """Exécute corrélation de Spearman + Chi-2"""
        data_clean = self.data[[var1, var2]].dropna()
        rho, p_spearman = spearmanr(data_clean[var1], data_clean[var2])
        
        contingence = pd.crosstab(self.data[var1], self.data[var2])
        chi2, p_chi2, dof, expected = chi2_contingency(contingence)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(contingence, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title(f'Table de contingence')
        
        axes[1].scatter(data_clean[var1], data_clean[var2], alpha=0.5)
        axes[1].set_xlabel(var1)
        axes[1].set_ylabel(var2)
        axes[1].set_title(f'Corrélation de Spearman\nρ = {rho:.3f}')
        
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Corrélation de Spearman + Chi-2',
            'rho': float(rho),
            'p_spearman': float(p_spearman),
            'chi2': float(chi2),
            'p_chi2': float(p_chi2),
            'ddl': int(dof),
            'graphique': graphique
        }
    
    def executer_ttest(self, var_cat, var_num):
        """Exécute le test t de Student"""
        groupes_uniques = self.data[var_cat].unique()
        groupe1 = self.data[self.data[var_cat] == groupes_uniques[0]][var_num].dropna()
        groupe2 = self.data[self.data[var_cat] == groupes_uniques[1]][var_num].dropna()
        
        t_stat, p_value = ttest_ind(groupe1, groupe2)
        
        plt.figure(figsize=(10, 6))
        self.data.boxplot(column=var_num, by=var_cat)
        plt.suptitle('')
        plt.title(f'Distribution de {var_num} par {var_cat}')
        plt.ylabel(var_num)
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Test t de Student (paramétrique)',
            'statistique': float(t_stat),
            'p_value': float(p_value),
            'moyenne_groupe1': float(groupe1.mean()),
            'moyenne_groupe2': float(groupe2.mean()),
            'groupe1': str(groupes_uniques[0]),
            'groupe2': str(groupes_uniques[1]),
            'graphique': graphique
        }
    
    def executer_mannwhitney(self, var_cat, var_num):
        """Exécute le test de Mann-Whitney U"""
        groupes_uniques = self.data[var_cat].unique()
        groupe1 = self.data[self.data[var_cat] == groupes_uniques[0]][var_num].dropna()
        groupe2 = self.data[self.data[var_cat] == groupes_uniques[1]][var_num].dropna()
        
        u_stat, p_value = mannwhitneyu(groupe1, groupe2, alternative='two-sided')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        self.data.boxplot(column=var_num, by=var_cat, ax=axes[0])
        axes[0].set_title(f'Boxplot: {var_num} par {var_cat}')
        axes[0].set_xlabel(var_cat)
        axes[0].set_ylabel(var_num)
        
        data_plot = self.data[[var_cat, var_num]].dropna()
        sns.violinplot(data=data_plot, x=var_cat, y=var_num, ax=axes[1])
        axes[1].set_title(f'Violin plot: {var_num} par {var_cat}')
        
        plt.suptitle('')
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Test de Mann-Whitney U',
            'statistique': float(u_stat),
            'p_value': float(p_value),
            'mediane_groupe1': float(groupe1.median()),
            'mediane_groupe2': float(groupe2.median()),
            'groupe1': str(groupes_uniques[0]),
            'groupe2': str(groupes_uniques[1]),
            'taille_effet': float(u_stat / (len(groupe1) * len(groupe2))),
            'graphique': graphique
        }
    
    def executer_anova(self, var_cat, var_num):
        """Exécute l'ANOVA"""
        groupes = [self.data[self.data[var_cat] == cat][var_num].dropna() 
                   for cat in self.data[var_cat].unique()]
        f_stat, p_value = f_oneway(*groupes)
        
        tukey_result = None
        if p_value < 0.05:
            data_clean = self.data[[var_cat, var_num]].dropna()
            tukey_result = str(pairwise_tukeyhsd(data_clean[var_num], data_clean[var_cat], alpha=0.05))
        
        plt.figure(figsize=(10, 6))
        self.data.boxplot(column=var_num, by=var_cat)
        plt.suptitle('')
        plt.title(f'Distribution de {var_num} par {var_cat}')
        plt.ylabel(var_num)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        moyennes_dict = {str(cat): float(self.data[self.data[var_cat] == cat][var_num].mean()) 
                        for cat in self.data[var_cat].unique()}
        
        return {
            'test': 'ANOVA',
            'statistique': float(f_stat),
            'p_value': float(p_value),
            'moyennes_par_groupe': moyennes_dict,
            'tukey': tukey_result,
            'graphique': graphique
        }
    
    def executer_kruskal(self, var_cat, var_num):
        """Exécute le test de Kruskal-Wallis"""
        groupes = [self.data[self.data[var_cat] == cat][var_num].dropna() 
                   for cat in self.data[var_cat].unique()]
        h_stat, p_value = kruskal(*groupes)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        self.data.boxplot(column=var_num, by=var_cat, ax=axes[0])
        axes[0].set_title(f'Boxplot: {var_num} par {var_cat}')
        axes[0].tick_params(axis='x', rotation=45)
        
        data_plot = self.data[[var_cat, var_num]].dropna()
        sns.violinplot(data=data_plot, x=var_cat, y=var_num, ax=axes[1])
        axes[1].set_title(f'Violin plot: {var_num} par {var_cat}')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('')
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        medianes = {str(cat): float(self.data[self.data[var_cat] == cat][var_num].median()) 
                    for cat in self.data[var_cat].unique()}
        
        return {
            'test': 'Test de Kruskal-Wallis',
            'statistique': float(h_stat),
            'p_value': float(p_value),
            'medianes_par_groupe': medianes,
            'graphique': graphique
        }
    
    def executer_correlation(self, var1, var2):
        """Exécute le test de corrélation de Pearson"""
        data_clean = self.data[[var1, var2]].dropna()
        r, p_value = pearsonr(data_clean[var1], data_clean[var2])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].scatter(data_clean[var1], data_clean[var2], alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(var1)
        axes[0].set_ylabel(var2)
        axes[0].set_title(f'Corrélation de Pearson\nr = {r:.3f}, p = {p_value:.4f}')
        
        z = np.polyfit(data_clean[var1], data_clean[var2], 1)
        p_poly = np.poly1d(z)
        axes[0].plot(data_clean[var1], p_poly(data_clean[var1]), "r--", alpha=0.8, linewidth=2)
        axes[0].grid(True, alpha=0.3)
        
        y_pred = p_poly(data_clean[var1])
        residus = data_clean[var2] - y_pred
        axes[1].scatter(y_pred, residus, alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Valeurs prédites')
        axes[1].set_ylabel('Résidus')
        axes[1].set_title('Graphique des résidus')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Corrélation de Pearson',
            'statistique': float(r),
            'p_value': float(p_value),
            'r_carre': float(r**2),
            'equation': f'y = {z[0]:.3f}x + {z[1]:.3f}',
            'graphique': graphique
        }
    
    def executer_spearman_kendall(self, var1, var2):
        """Exécute les corrélations de Spearman et Kendall"""
        data_clean = self.data[[var1, var2]].dropna()
        
        rho_spearman, p_spearman = spearmanr(data_clean[var1], data_clean[var2])
        tau_kendall, p_kendall = kendalltau(data_clean[var1], data_clean[var2])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].scatter(data_clean[var1], data_clean[var2], alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(var1)
        axes[0].set_ylabel(var2)
        axes[0].set_title(f'Corrélation de Spearman\nρ = {rho_spearman:.3f}, p = {p_spearman:.4f}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(data_clean[var1], data_clean[var2], alpha=0.6, 
                       edgecolors='black', linewidth=0.5, color='orange')
        axes[1].set_xlabel(var1)
        axes[1].set_ylabel(var2)
        axes[1].set_title(f'Corrélation de Kendall\nτ = {tau_kendall:.3f}, p = {p_kendall:.4f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        graphique = self.plot_to_base64()
        
        return {
            'test': 'Corrélations de Spearman et Kendall',
            'rho_spearman': float(rho_spearman),
            'p_spearman': float(p_spearman),
            'tau_kendall': float(tau_kendall),
            'p_kendall': float(p_kendall),
            'graphique': graphique
        }
    
    def analyser(self):
        """Effectue l'analyse statistique complète"""
        self.resultats = []
        variables_list = list(self.variables.keys())
        
        for i in range(len(variables_list)):
            for j in range(i+1, len(variables_list)):
                var1, var2 = variables_list[i], variables_list[j]
                type1, type2 = self.variables[var1], self.variables[var2]
                
                type_test, nom_test = self.determiner_test(var1, var2, type1, type2)
                
                if type_test is None:
                    continue
                
                try:
                    if type_test == 'chi2':
                        resultat = self.executer_chi2(var1, var2)
                    elif type_test == 'fisher':
                        resultat = self.executer_fisher(var1, var2)
                    elif type_test == 'spearman_ordinal':
                        resultat = self.executer_spearman_ordinal(var1, var2)
                    elif type_test == 'ttest':
                        var_cat = var1 if type1 in ['nominale', 'ordinale'] else var2
                        var_num = var2 if var_cat == var1 else var1
                        resultat = self.executer_ttest(var_cat, var_num)
                    elif type_test == 'mannwhitney':
                        var_cat = var1 if type1 in ['nominale', 'ordinale'] else var2
                        var_num = var2 if var_cat == var1 else var1
                        resultat = self.executer_mannwhitney(var_cat, var_num)
                    elif type_test == 'anova':
                        var_cat = var1 if type1 in ['nominale', 'ordinale'] else var2
                        var_num = var2 if var_cat == var1 else var1
                        resultat = self.executer_anova(var_cat, var_num)
                    elif type_test == 'kruskal':
                        var_cat = var1 if type1 in ['nominale', 'ordinale'] else var2
                        var_num = var2 if var_cat == var1 else var1
                        resultat = self.executer_kruskal(var_cat, var_num)
                    elif type_test == 'pearson':
                        resultat = self.executer_correlation(var1, var2)
                    elif type_test == 'spearman_kendall':
                        resultat = self.executer_spearman_kendall(var1, var2)
                    
                    resultat['var1'] = var1
                    resultat['var2'] = var2
                    self.resultats.append(resultat)
                    
                except Exception as e:
                    print(f"Erreur: {e}")
        
        return self.resultats

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Aucun fichier fourni'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nom de fichier vide'})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        analyseur = AnalyseurStatistiqueWeb()
        success, result = analyseur.charger_donnees(filepath)
        
        if success:
            # Sauvegarder l'analyseur en session (simplifié ici)
            return jsonify({
                'success': True,
                'columns': result,
                'rows': len(analyseur.data),
                'filepath': filepath
            })
        else:
            return jsonify({'success': False, 'error': result})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    filepath = data.get('filepath')
    variables_config = data.get('variables')
    
    analyseur = AnalyseurStatistiqueWeb()
    success, columns = analyseur.charger_donnees(filepath)
    
    if not success:
        return jsonify({'success': False, 'error': columns})
    
    analyseur.definir_variables(variables_config)
    resultats = analyseur.analyser()
    
    return jsonify({
        'success': True,
        'resultats': resultats
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)