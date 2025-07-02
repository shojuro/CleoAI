{{/*
Expand the name of the chart.
*/}}
{{- define "cleoai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "cleoai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cleoai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cleoai.labels" -}}
helm.sh/chart: {{ include "cleoai.chart" . }}
{{ include "cleoai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cleoai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cleoai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "cleoai.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "cleoai.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URLs
*/}}
{{- define "cleoai.postgresqlUrl" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s-postgresql:%d/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "cleoai.fullname" .) (.Values.postgresql.service.port | int) .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.api.secrets.database.postgresUrl }}
{{- end }}
{{- end }}

{{- define "cleoai.mongodbUrl" -}}
{{- if .Values.mongodb.enabled }}
{{- printf "mongodb://%s:%s@%s-mongodb:%d/%s" .Values.mongodb.auth.username .Values.mongodb.auth.password (include "cleoai.fullname" .) (.Values.mongodb.service.port | int) .Values.mongodb.auth.database }}
{{- else }}
{{- .Values.api.secrets.database.mongodbUrl }}
{{- end }}
{{- end }}

{{- define "cleoai.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://:%s@%s-redis-master:%d/0" .Values.redis.auth.password (include "cleoai.fullname" .) (.Values.redis.master.service.port | int) }}
{{- else }}
{{- .Values.api.secrets.database.redisUrl }}
{{- end }}
{{- end }}