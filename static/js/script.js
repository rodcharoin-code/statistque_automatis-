let currentFilepath = null;
let columnsData = [];

// Upload du fichier
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Veuillez sélectionner un fichier');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    document.getElementById('uploadProgress').style.display = 'block';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentFilepath = result.filepath;
            columnsData = result.columns;
            
            document.getElementById('uploadProgress').style.display = 'none';
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileDetails').innerHTML = `
                <strong>Nombre de lignes:</strong> ${result.rows}<br>
                <strong>Nombre de colonnes:</strong> ${result.columns.length}<br>
                <strong>Colonnes:</strong> ${result.columns.join(', ')}
            `;
            
            // Afficher l'étape 2
            document.getElementById('step2').style.display = 'block';
            generateVariableInputs(result.columns);
            
            // Scroll vers l'étape 2
            document.getElementById('step2').scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Erreur: ' + result.error);
        }
    } catch (error) {
        alert('Erreur lors du chargement: ' + error);
    }
});

// Générer les inputs pour les variables
function generateVariableInputs(columns) {
    const container = document.getElementById('variablesContainer');
    container.innerHTML = '';
    
    columns.forEach(column => {
        const div = document.createElement('div');
        div.className = 'variable-item';
        div.innerHTML = `
            <div class="row align-items-center">
                <div class="col-md-4">
                    <label class="form-label fw-bold">${column}</label>
                </div>
                <div class="col-md-8">
                    <select class="form-select variable-type" data-column="${column}">
                        <option value="">-- Sélectionnez le type --</option>
                        <option value="nominale">Nominale (catégorielle sans ordre)</option>
                        <option value="ordinale">Ordinale (catégorielle avec ordre)</option>
                        <option value="discrète">Discrète (numérique entière)</option>
                        <option value="continue">Continue (numérique décimale)</option>
                    </select>
                </div>
            </div>
        `;
        container.appendChild(div);
    });
}

// Lancer l'analyse
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const variableSelects = document.querySelectorAll('.variable-type');
    const variables = {};
    let allSelected = true;
    
    variableSelects.forEach(select => {
        const column = select.dataset.column;
        const type = select.value;
        
        if (!type) {
            allSelected = false;
        } else {
            variables[column] = type;
        }
    });
    
    if (!allSelected) {
        alert('Veuillez définir le type de toutes les variables');
        return;
    }
    
    // Afficher l'étape 3
    document.getElementById('step3').style.display = 'block';
    document.getElementById('analysisProgress').style.display = 'block';
    document.getElementById('step3').scrollIntoView({ behavior: 'smooth' });
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: currentFilepath,
                variables: variables
            })
        });
        
        const result = await response.json();
        
        document.getElementById('analysisProgress').style.display = 'none';
        
        if (result.success) {
            displayResults(result.resultats);
            document.getElementById('downloadReportBtn').style.display = 'block';
        } else {
            alert('Erreur lors de l\'analyse: ' + result.error);
        }
    } catch (error) {
        alert('Erreur: ' + error);
        document.getElementById('analysisProgress').style.display = 'none';
    }
});

// Afficher les résultats
function displayResults(resultats) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    if (resultats.length === 0) {
        container.innerHTML = '<div class="alert alert-warning">Aucun résultat à afficher</div>';
        return;
    }
    
    resultats.forEach((res, index) => {
        const isSignificant = res.p_value < 0.05;
        const significanceClass = isSignificant ? 'significant' : 'not-significant';
        const significanceIcon = isSignificant ? 'check-circle-fill' : 'exclamation-triangle-fill';
        const significanceColor = isSignificant ? 'success' : 'warning';
        
        let interpretation = generateInterpretation(res, isSignificant);
        
        const resultCard = `
            <div class="card result-card">
                <div class="card-body">
                    <h5 class="card-title">
                        <i class="bi bi-graph-up-arrow"></i> 
                        Analyse ${index + 1}: ${res.var1} vs ${res.var2}
                    </h5>
                    <hr>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <span class="badge bg-primary stat-badge">Test: ${res.test}</span>
                        </div>
                        <div class="col-md-6 text-end">
                            <span class="badge bg-${significanceColor} stat-badge">
                                <i class="bi bi-${significanceIcon}"></i>
                                ${isSignificant ? 'Significatif' : 'Non significatif'}
                            </span>
                        </div>
                    </div>
                    
                    <div class="statistics mb-3">
                        ${generateStatistics(res)}
                    </div>
                    
                    ${res.graphique ? `<img src="${res.graphique}" class="result-graph" alt="Graphique">` : ''}
                    
                    <div class="interpretation ${significanceClass}">
                        <h6><i class="bi bi-lightbulb"></i> Interprétation</h6>
                        ${interpretation}
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML += resultCard;
    });
}

// Générer les statistiques selon le test
function generateStatistics(res) {
    let html = '<table class="table table-sm table-borderless">';
    
    if (res.test === 'Chi-2') {
        html += `
            <tr><td><strong>χ²:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>Degrés de liberté:</strong></td><td>${res.ddl}</td></tr>
            <tr><td><strong>Valeur p:</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
        `;
        if (res.effectif_theorique_min < 5) {
            html += '<tr><td colspan="2"><span class="badge bg-warning">⚠ Effectifs théoriques < 5</span></td></tr>';
        }
    }
    
    else if (res.test === 'Test exact de Fisher') {
        html += `
            <tr><td><strong>Odds Ratio:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>Valeur p (exacte):</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
        `;
    }
    
    else if (res.test === 'Corrélation de Spearman + Chi-2') {
        html += `
            <tr><td><strong>ρ (Spearman):</strong></td><td>${res.rho.toFixed(3)}</td></tr>
            <tr><td><strong>p (Spearman):</strong></td><td>${res.p_spearman.toFixed(4)}</td></tr>
            <tr><td><strong>χ²:</strong></td><td>${res.chi2.toFixed(3)}</td></tr>
            <tr><td><strong>p (Chi-2):</strong></td><td>${res.p_chi2.toFixed(4)}</td></tr>
        `;
    }
    
    else if (res.test === 'Test t de Student (paramétrique)') {
        html += `
            <tr><td><strong>Statistique t:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>Valeur p:</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
            <tr><td><strong>Moyenne ${res.groupe1}:</strong></td><td>${res.moyenne_groupe1.toFixed(3)}</td></tr>
            <tr><td><strong>Moyenne ${res.groupe2}:</strong></td><td>${res.moyenne_groupe2.toFixed(3)}</td></tr>
            <tr><td><strong>Différence:</strong></td><td>${Math.abs(res.moyenne_groupe1 - res.moyenne_groupe2).toFixed(3)}</td></tr>
        `;
    }
    
    else if (res.test === 'Test de Mann-Whitney U') {
        html += `
            <tr><td><strong>Statistique U:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>Valeur p:</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
            <tr><td><strong>Médiane ${res.groupe1}:</strong></td><td>${res.mediane_groupe1.toFixed(3)}</td></tr>
            <tr><td><strong>Médiane ${res.groupe2}:</strong></td><td>${res.mediane_groupe2.toFixed(3)}</td></tr>
            <tr><td><strong>Taille d'effet:</strong></td><td>${res.taille_effet.toFixed(3)}</td></tr>
        `;
    }
    
    else if (res.test === 'ANOVA') {
        html += `
            <tr><td><strong>Statistique F:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>Valeur p:</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
        `;
        for (let [groupe, moyenne] of Object.entries(res.moyennes_par_groupe)) {
            html += `<tr><td><strong>Moyenne ${groupe}:</strong></td><td>${moyenne.toFixed(3)}</td></tr>`;
        }
    }
    
    else if (res.test === 'Test de Kruskal-Wallis') {
        html += `
            <tr><td><strong>Statistique H:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>Valeur p:</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
        `;
        for (let [groupe, mediane] of Object.entries(res.medianes_par_groupe)) {
            html += `<tr><td><strong>Médiane ${groupe}:</strong></td><td>${mediane.toFixed(3)}</td></tr>`;
        }
    }
    
    else if (res.test === 'Corrélation de Pearson') {
        html += `
            <tr><td><strong>Coefficient r:</strong></td><td>${res.statistique.toFixed(3)}</td></tr>
            <tr><td><strong>R²:</strong></td><td>${res.r_carre.toFixed(3)}</td></tr>
            <tr><td><strong>Valeur p:</strong></td><td>${res.p_value.toFixed(4)}</td></tr>
            <tr><td><strong>Équation:</strong></td><td>${res.equation}</td></tr>
        `;
    }
    
    else if (res.test === 'Corrélations de Spearman et Kendall') {
        html += `
            <tr><td><strong>ρ (Spearman):</strong></td><td>${res.rho_spearman.toFixed(3)}</td></tr>
            <tr><td><strong>p (Spearman):</strong></td><td>${res.p_spearman.toFixed(4)}</td></tr>
            <tr><td><strong>τ (Kendall):</strong></td><td>${res.tau_kendall.toFixed(3)}</td></tr>
            <tr><td><strong>p (Kendall):</strong></td><td>${res.p_kendall.toFixed(4)}</td></tr>
        `;
    }
    
    html += '</table>';
    return html;
}

// Générer l'interprétation
function generateInterpretation(res, isSignificant) {
    let interpretation = '';
    const alpha = 'α = 0.05';
    
    if (res.test.includes('Chi-2') || res.test.includes('Fisher')) {
        interpretation = `L'association entre <strong>${res.var1}</strong> et <strong>${res.var2}</strong> est <strong>${isSignificant ? 'significative' : 'non significative'}</strong> (${alpha}).`;
    }
    
    else if (res.test.includes('Spearman') && res.test.includes('ordinal')) {
        const force = Math.abs(res.rho) < 0.3 ? 'faible' : (Math.abs(res.rho) < 0.7 ? 'modérée' : 'forte');
        const sens = res.rho > 0 ? 'positive' : 'négative';
        interpretation = `Il existe une corrélation <strong>${sens} ${force}</strong> entre les variables (ρ = ${res.rho.toFixed(3)}). `;
        interpretation += `Cette corrélation est <strong>${res.p_spearman < 0.05 ? 'significative' : 'non significative'}</strong>. `;
        interpretation += `L'association globale (Chi-2) est <strong>${res.p_chi2 < 0.05 ? 'significative' : 'non significative'}</strong>.`;
    }
    
    else if (res.test.includes('Student') || res.test.includes('Mann-Whitney') || res.test.includes('ANOVA') || res.test.includes('Kruskal')) {
        interpretation = `La différence entre les groupes est <strong>${isSignificant ? 'significative' : 'non significative'}</strong> (${alpha}). `;
        if (res.test.includes('Mann-Whitney') || res.test.includes('Kruskal')) {
            interpretation += '<br><em>Note: Test non-paramétrique utilisé (données non normales ou variances hétérogènes).</em>';
        }
    }
    
    else if (res.test.includes('Pearson')) {
        const force = Math.abs(res.statistique) < 0.3 ? 'faible' : (Math.abs(res.statistique) < 0.7 ? 'modérée' : 'forte');
        const sens = res.statistique > 0 ? 'positive' : 'négative';
        interpretation = `Il existe une corrélation <strong>${sens} ${force}</strong> entre <strong>${res.var1}</strong> et <strong>${res.var2}</strong> (r = ${res.statistique.toFixed(3)}). `;
        interpretation += `Cette corrélation est <strong>${isSignificant ? 'significative' : 'non significative'}</strong> (${alpha}). `;
        interpretation += `<br><strong>${(res.r_carre * 100).toFixed(1)}%</strong> de la variance est expliquée.`;
    }
    
    else if (res.test.includes('Spearman') && res.test.includes('Kendall')) {
        const force = Math.abs(res.rho_spearman) < 0.3 ? 'faible' : (Math.abs(res.rho_spearman) < 0.7 ? 'modérée' : 'forte');
        const sens = res.rho_spearman > 0 ? 'positive' : 'négative';
        interpretation = `Il existe une relation <strong>${sens} ${force}</strong> entre les variables. `;
        interpretation += `Les deux tests confirment une corrélation <strong>${res.p_spearman < 0.05 ? 'significative' : 'non significative'}</strong>. `;
        interpretation += '<br><em>Note: Tests non-paramétriques utilisés (données non normales).</em>';
    }
    
    return interpretation;
}

// Télécharger le rapport
document.getElementById('downloadReportBtn').addEventListener('click', () => {
    alert('Fonctionnalité de téléchargement en développement');
});
