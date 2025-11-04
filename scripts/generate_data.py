#!/usr/bin/env python3
"""
Script para gerar dataset sintético realista de Angola
Grupo 7 - Projeto Capstone: Sistema de Previsão de Inundações

Gera dados de:
- Regiões (Luanda, Benguela, Huambo)
- Hierarquia geográfica (bairros, comunas, distritos, municípios)
- Dados meteorológicos (chuva, temperatura, humidade, pressão)
- Histórico de inundações
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Definir seed para reprodutibilidade
np.random.seed(42)

# Estrutura geográfica realista de Angola
GEOGRAPHIC_HIERARCHY = {
    "Luanda": {
        "province": "Luanda",
        "districts": {
            "Distrito 1": {
                "communes": {
                    "Comuna Benilson": {
                        "neighborhoods": ["Malueca", "Cazenga", "Rangel", "Samba"]
                    },
                    "Comuna Kilamba": {
                        "neighborhoods": ["Kilamba Kiaxi", "Viana", "Cacuaco"]
                    }
                }
            },
            "Distrito 2": {
                "communes": {
                    "Comuna Icolo": {
                        "neighborhoods": ["Icolo e Bengo", "Belas", "Zango"]
                    }
                }
            }
        }
    },
    "Benguela": {
        "province": "Benguela",
        "districts": {
            "Distrito 1": {
                "communes": {
                    "Comuna Benguela": {
                        "neighborhoods": ["Lobito", "Catumbela", "Baia Farta"]
                    },
                    "Comuna Balombo": {
                        "neighborhoods": ["Balombo", "Ganda"]
                    }
                }
            }
        }
    },
    "Huambo": {
        "province": "Huambo",
        "districts": {
            "Distrito 1": {
                "communes": {
                    "Comuna Huambo": {
                        "neighborhoods": ["Huambo Centro", "Ekundu", "Caala"]
                    },
                    "Comuna Ebo": {
                        "neighborhoods": ["Ebo", "Mungo"]
                    }
                }
            }
        }
    }
}

def generate_regions_data():
    """Gera dados de regiões com hierarquia geográfica"""
    regions = []
    region_id = 1
    
    for region_name, region_data in GEOGRAPHIC_HIERARCHY.items():
        for district_name, district_data in region_data["districts"].items():
            for commune_name, commune_data in district_data["communes"].items():
                for neighborhood in commune_data["neighborhoods"]:
                    # Coordenadas aproximadas de Angola
                    if region_name == "Luanda":
                        lat = np.random.uniform(-8.9, -8.8)
                        lon = np.random.uniform(13.2, 13.3)
                        altitude = np.random.randint(5, 50)
                    elif region_name == "Benguela":
                        lat = np.random.uniform(-12.6, -12.5)
                        lon = np.random.uniform(13.3, 13.5)
                        altitude = np.random.randint(10, 100)
                    else:  # Huambo
                        lat = np.random.uniform(-12.8, -12.7)
                        lon = np.random.uniform(15.7, 15.9)
                        altitude = np.random.randint(1600, 1800)
                    
                    regions.append({
                        "id": region_id,
                        "region_name": region_name,
                        "province": region_data["province"],
                        "district": district_name,
                        "commune": commune_name,
                        "neighborhood": neighborhood,
                        "latitude": round(lat, 4),
                        "longitude": round(lon, 4),
                        "altitude": altitude,
                        "risk_level": np.random.choice(["low", "medium", "high", "critical"]),
                        "population": np.random.randint(1000, 50000),
                        "created_at": datetime.now().isoformat()
                    })
                    region_id += 1
    
    return pd.DataFrame(regions)

def generate_weather_data(num_records=1000):
    """Gera dados meteorológicos sintéticos realistas"""
    weather_data = []
    
    regions_df = generate_regions_data()
    region_ids = regions_df["id"].tolist()
    
    for _ in range(num_records):
        region_id = np.random.choice(region_ids)
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        # Dados realistas para Angola
        temperature = np.random.normal(25, 5)  # Média 25°C
        humidity = np.random.normal(70, 15)  # Média 70%
        precipitation = np.random.exponential(5)  # Distribuição exponencial (mais dias sem chuva)
        pressure = np.random.normal(1013, 5)  # Pressão normal
        wind_speed = np.random.gamma(2, 2)  # Velocidade do vento
        
        weather_data.append({
            "id": len(weather_data) + 1,
            "region_id": region_id,
            "timestamp": timestamp.isoformat(),
            "temperature_celsius": round(max(-10, min(50, temperature)), 2),
            "humidity_percent": round(max(0, min(100, humidity)), 2),
            "precipitation_mm": round(max(0, precipitation), 2),
            "atmospheric_pressure_mb": round(pressure, 2),
            "wind_speed_kmh": round(max(0, wind_speed), 2),
            "source": "Synthetic Data - Grupo 7",
            "created_at": datetime.now().isoformat()
        })
    
    return pd.DataFrame(weather_data)

def generate_historical_events(num_events=100):
    """Gera histórico de eventos de inundação"""
    events = []
    
    regions_df = generate_regions_data()
    region_ids = regions_df["id"].tolist()
    
    for i in range(num_events):
        region_id = np.random.choice(region_ids)
        event_date = datetime.now() - timedelta(days=np.random.randint(30, 3650))
        
        events.append({
            "id": i + 1,
            "region_id": region_id,
            "event_date": event_date.isoformat(),
            "severity": np.random.choice(["minor", "moderate", "severe", "catastrophic"]),
            "affected_population": np.random.randint(100, 10000),
            "damage_estimate_aoa": np.random.randint(100000, 10000000),  # Kwanza Angolano
            "description": f"Evento de inundação simulado - Grupo 7",
            "data_source": "Synthetic Data - Grupo 7",
            "created_at": datetime.now().isoformat()
        })
    
    return pd.DataFrame(events)

def generate_predictions(num_predictions=500):
    """Gera previsões de inundação"""
    predictions = []
    
    regions_df = generate_regions_data()
    region_ids = regions_df["id"].tolist()
    
    for i in range(num_predictions):
        region_id = np.random.choice(region_ids)
        probability = np.random.beta(2, 5)  # Distribuição beta (mais previsões com baixa probabilidade)
        
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "medium"
        elif probability < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        predictions.append({
            "id": i + 1,
            "region_id": region_id,
            "prediction_time": datetime.now().isoformat(),
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "confidence": round(np.random.uniform(0.7, 0.99), 4),
            "model_version": "v1.0.0 - Grupo 7",
            "created_at": datetime.now().isoformat()
        })
    
    return pd.DataFrame(predictions)

def main():
    """Gera todos os datasets e salva em CSV e Excel"""
    
    print("🌊 Gerando dataset sintético realista de Angola - Grupo 7")
    print("=" * 60)
    
    # Gerar dados
    print("📍 Gerando dados de regiões e hierarquia geográfica...")
    regions_df = generate_regions_data()
    print(f"   ✓ {len(regions_df)} regiões/bairros/comunas/distritos criados")
    
    print("🌤️  Gerando dados meteorológicos...")
    weather_df = generate_weather_data(1000)
    print(f"   ✓ {len(weather_df)} registros meteorológicos criados")
    
    print("📚 Gerando histórico de eventos...")
    events_df = generate_historical_events(100)
    print(f"   ✓ {len(events_df)} eventos históricos criados")
    
    print("🔮 Gerando previsões...")
    predictions_df = generate_predictions(500)
    print(f"   ✓ {len(predictions_df)} previsões criadas")
    
    # Criar diretório de dados se não existir
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Salvar em CSV
    print("\n💾 Salvando em CSV...")
    regions_df.to_csv(f"{data_dir}/regions.csv", index=False, encoding='utf-8')
    print(f"   ✓ regions.csv salvo")
    
    weather_df.to_csv(f"{data_dir}/weather_data.csv", index=False, encoding='utf-8')
    print(f"   ✓ weather_data.csv salvo")
    
    events_df.to_csv(f"{data_dir}/historical_events.csv", index=False, encoding='utf-8')
    print(f"   ✓ historical_events.csv salvo")
    
    predictions_df.to_csv(f"{data_dir}/predictions.csv", index=False, encoding='utf-8')
    print(f"   ✓ predictions.csv salvo")
    
    # Salvar em Excel
    print("\n📊 Salvando em Excel...")
    excel_file = f"{data_dir}/flood_prediction_dataset_grupo7.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        regions_df.to_excel(writer, sheet_name='Regions', index=False)
        weather_df.to_excel(writer, sheet_name='Weather Data', index=False)
        events_df.to_excel(writer, sheet_name='Historical Events', index=False)
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
    
    print(f"   ✓ flood_prediction_dataset_grupo7.xlsx salvo")
    
    # Estatísticas
    print("\n📈 Estatísticas dos Dados:")
    print(f"   Regiões: {len(regions_df)}")
    print(f"   Registros meteorológicos: {len(weather_df)}")
    print(f"   Eventos históricos: {len(events_df)}")
    print(f"   Previsões: {len(predictions_df)}")
    print(f"\n   Temperatura média: {weather_df['temperature_celsius'].mean():.1f}°C")
    print(f"   Humidade média: {weather_df['humidity_percent'].mean():.1f}%")
    print(f"   Precipitação média: {weather_df['precipitation_mm'].mean():.2f}mm")
    print(f"   Pressão média: {weather_df['atmospheric_pressure_mb'].mean():.2f}mb")
    
    print("\n✅ Dataset sintético gerado com sucesso!")
    print(f"   Localização: {data_dir}/")

if __name__ == "__main__":
    main()
