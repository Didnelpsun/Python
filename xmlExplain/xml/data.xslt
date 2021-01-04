<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:fn="http://www.w3.org/2005/xpath-functions">
	<!--因为需要以表格形式呈现，所以输出为HTML文件格式-->
	<!--编码形式采取gb2312，否则会乱码-->
	<xsl:output method="html" encoding="gb2312" indent="yes"/>
	<xsl:template match= "/">
		<html>
			<head>
				<title>学生信息</title>
			</head>
			<body>
				<table border= "1" style="border-collapse:collapse; padding:1px;">
					<tbody>
						<tr>
							<th style="width:200px;">宿舍号</th>
							<th style="width:200px;">学号</th>
							<th style="width:200px;">姓名</th>
							<th style="width:200px;">电话号码</th>
							<th style="width:400px;">备注</th>
						</tr>
						<xsl:apply-templates select= "//dorm" />
					</tbody>
				</table>
			</body>
		</html>
	</xsl:template >
	<xsl:template match="//dorm">
		<xsl:call-template name="value_first">
			<xsl:with-param name="student" select="student[1]"></xsl:with-param>
		</xsl:call-template>
		<xsl:call-template name="value">
			<xsl:with-param name="student" select="student[2]"></xsl:with-param>
		</xsl:call-template>
		<xsl:call-template name="value">
			<xsl:with-param name="student" select="student[3]"></xsl:with-param>
		</xsl:call-template>
		<xsl:call-template name="value">
			<xsl:with-param name="student" select="student[4]"></xsl:with-param>
		</xsl:call-template>
	</xsl:template>
	<xsl:template name="value_first">
		<xsl:param name="student"/>
		<tr>
			<td align="center" rowspan="4"><xsl:value-of select ="$ student/parent::dorm/@id"/></td>
			<td align="center"><xsl:value-of select ="$ student/@id" /></td>
			<td align="center"><xsl:value-of select= " $ student/name"/></td>
			<td align="center"><xsl:value-of select = "$ student/telephone" /></td>
			<td align="center"><xsl:value-of select = "$ student/remarks"/></td>
		</tr>
	</xsl:template>
	<xsl:template name="value">
		<xsl:param name="student"/>
		<tr>
			<td align="center"><xsl:value-of select ="$ student/@id" /></td>
			<td align="center"><xsl:value-of select= " $ student/name"/></td>
			<td align="center"><xsl:value-of select = "$ student/telephone" /></td>
			<td align="center"><xsl:value-of select = "$ student/remarks"/></td>
		</tr>
	</xsl:template>
</xsl:stylesheet>
