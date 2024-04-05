import streamlit as st
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from seaborn.palettes import blend_palette

st.title('Simulação de Transferência de Calor em um Trocador de Calor Tubo Duplo')
st.write('Este site simula um trocador de calor tubo duplo operando em correntes paralelas. Ao rodar a simulação, você poderá visualizar o perfil de temperatura dos fluidos 1 (frio) e 2 (quente) conforme o tempo passa. Você também poderá visualizar o gráfico de variação das temperaturas dos fluidos 1 e 2 quando o trocador atinge o regime permanente.')
st.write('Ao final da página, você também encontrará um botão que roda a simulação com um exemplo padrão.')

# Carregar a imagem
st.image('Caso 2.png', use_column_width=True)

# Criando a figura para o gráfico em regime permanente
fig_permanente = plt.figure(figsize=(8, 6))

def run_simulation(L, r1, r2, n, m1, Cp1, rho1, m2, Cp2, rho2, T1i, T2i, T0, U, dx, t_final, dt):
    Ac1 = np.pi * r1**2
    Ac2 = np.pi * (r2**2-r1**2)

    x = np.linspace(dx/2, L-dx/2, n)
    T1 = np.ones(n) * T1i
    T2 = np.ones(n) * T2i
    t = np.arange(0, t_final, dt)
    
    # Função que define a EDO para a variação da temperatura para o Fluido 1
    def dT1dt_function(T1, t):
        dT1dt = np.zeros(n)
        dT1dt[1:n] = (m1 * Cp1 * (T1[0:n-1] - T1[1:n]) + U * 2 * np.pi * r1 * dx * (T2[1:n] - T1[1:n])) / (rho1 * Cp1 * dx * Ac1)
        dT1dt[0] = (m1 * Cp1 * (T1i - T1[0]) + U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1)
        return dT1dt
    
    # Função que define a EDO para a variação da temperatura para o Fluido 2
    def dT2dt_function(T2, t):
        dT2dt = np.zeros(n)
        dT2dt[1:n] = (m2 * Cp2 * (T2[0:n-1] - T2[1:n]) - U * 2 * np.pi * r1 * dx * (T2[1:n] - T1[1:n])) / (rho2 * Cp2 * dx * Ac2)
        dT2dt[0] = (m2 * Cp2 * (T2i - T2[0]) - U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho2 * Cp2 * dx * Ac2)
        return dT2dt
    
    T_out1 = odeint(dT1dt_function, T1, t)
    T_out1 = T_out1 - 273.15
    T_out2 = odeint(dT2dt_function, T2, t)
    T_out2 = T_out2 - 273.15
    
    # Criação dos DataFrames
    df_Temp1 = pd.DataFrame(np.array(T_out1), columns=x)
    df_Temp2 = pd.DataFrame(np.array(T_out2), columns=x)
    
    # Criando as paletas de cores para os fluidos 1 e 2
    paleta_calor = blend_palette(['blue', 'yellow', 'orange','red'], as_cmap=True, n_colors=100)
    
    # Função que atualiza o plot para o Fluido 1
    def update_plot1(t):
        plt.clf()
        line = pd.DataFrame(df_Temp1.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Tempo: {t} (s)')
        
    # Função que atualiza o plot para o Fluido 2
    def update_plot2(t):
        plt.clf()
        line = pd.DataFrame(df_Temp2.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Tempo: {t} (s)')

    # Criação e exibição da figura 1
    fig_ani1 = plt.figure(figsize=(8,6))
    ani1 = FuncAnimation(fig_ani1, update_plot1, frames=df_Temp1.shape[0], repeat=False)
    save1 = ani1.save('Variação da Temperatura - Fluido 1.gif', writer='pillow', fps=10)
    
    # Criação e exibição da figura 2
    fig_ani2 = plt.figure(figsize=8,6))
    ani2 = FuncAnimation(fig_ani2, update_plot2, frames=df_Temp2.shape[0], repeat=False)
    save2 = ani2.save('Variação da Temperatura - Fluido 2.gif', writer='pillow', fps=10)
    
    # Exibindo a simulação
    with st.expander("Visualização da Simulação em tempo real para o Fluido 1 (frio) (Clique aqui para ver)"):
        st.image('Variação da Temperatura - Fluido 1.gif')
    with st.expander("Visualização da Simulação em tempo real para o Fluido 2 (quente) (Clique aqui para ver)"):
        st.image('Variação da Temperatura - Fluido 2.gif')
    
    # Inicialização de variáveis
    t_teste = dt
    T_permanente_anterior1 = np.inf
    T_permanente_atual1 = np.zeros(n)
    T_permanente_anterior2 = np.inf
    T_permanente_atual2 = np.zeros(n)

    # Loop para testar diferentes tempos até que o regime permanente seja alcançado para ambos os fluidos
    while np.max(np.abs(T_permanente_atual1 - T_permanente_anterior1)) > 0.01 or np.max(np.abs(T_permanente_atual2 - T_permanente_anterior2)) > 0.01:  # Critério de convergência
        T_permanente_anterior1 = T_permanente_atual1.copy()
        T_permanente_anterior2 = T_permanente_atual2.copy()
        
        # Resolvendo a EDO usando o odeint para o regime permanente para o Fluido 1
        T_permanente1 = odeint(dT1dt_function, T1, [0, t_teste])[-1]
        T_permanente_atual1 = T_permanente1
        
        # Resolvendo a EDO usando o odeint para o regime permanente para o Fluido 2
        T_permanente2 = odeint(dT2dt_function, T2, [0, t_teste])[-1]
        T_permanente_atual2 = T_permanente2

        # Incrementando o tempo de teste
        t_teste += dt
        
    # Exibindo o gráfico de variação da temperatura ao longo do comprimento em regime permanente para ambos os fluidos
    plt.figure(fig_permanente)
    plt.plot(x, T_permanente_atual1 - 273.15, color='blue', label='Fluido 1')
    plt.plot(x, T_permanente_atual2 - 273.15, color='red', label='Fluido 2')
    plt.xlabel('Comprimento (m)')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    st.pyplot(plt)

st.title('Parâmetros de Entrada')
# Valores input
L = st.number_input('Comprimento do tubo (m)', min_value=0.0)
r1 = st.number_input('Raio interno do tubo (m)', min_value=0.0)
r2 = st.number_input('Raio externo do tubo (m)', min_value=0.0)
n = st.number_input('Número de nós para a discretização', min_value=1)
m1 = st.number_input('Vazão Mássica do Fluido 1 (kg/s)', min_value=0.0)
Cp1 = st.number_input('Capacidade de Calor Específico do Fluido 1 (J/kg.K)', min_value=0.0)
rho1 = st.number_input('Massa Específica do Fluido 1 (kg/m³)', min_value=0.0)
m2 = st.number_input('Vazão Mássica do Fluido 2 (kg/s)', min_value=0.0)
Cp2 = st.number_input('Capacidade de Calor Específico do Fluido 2 (J/kg.K)', min_value=0.0)
rho2 = st.number_input('Massa Específica do Fluido 2 (kg/m³)', min_value=0.0)
T1i = st.number_input('Temperatura de entrada do fluido 1 no trocador (K)')
T2i = st.number_input('Temperatura de entrada do fluido 2 no trocador (K)')
T0 = st.number_input('Temperatura inicial do tubo (K)')
U = st.number_input('Coeficiente Global de Transferência de Calor (W/m².K)', min_value=0.0)
dx = L / n

t_final = st.number_input('Tempo de Simulação (s)', min_value=0.0)
dt = st.number_input('Passo de Tempo (s)', min_value=0.0)

if st.button('Rodar Simulação'):
    run_simulation(L, r1, r2, n, m1, Cp1, rho1, m2, Cp2, rho2, T1i, T2i, T0, U, dx, t_final, dt)
elif st.button('Rodar Exemplo Padrão'):
    run_simulation(50, 0.1, 0.15, 100, 3, 4180, 1000, 5, 4180, 1000, 400, 800, 300, 340, 50 / 100, 700, 1)
