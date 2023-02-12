from numpy import empty
import xlwings as xw
import os
import sys

# save training/validation results : save model only after all models are finished
def save_train_results_10_times(excel_path, worksheet, model_name_list, training_time_list, best_val_acc_list):
  app = xw.App(visible = True,add_book = False)  
  workbook = app.books.open(excel_path)
  worksheet = workbook.sheets(worksheet)
  j = 2
  for i in range(len(model_name_list)):
    worksheet.range('A'+str(j)).options(transpose=True).value = model_name_list[i]
    worksheet.range('B'+str(j)).options(transpose=True).value = training_time_list[i]
    worksheet.range('C'+str(j)).options(transpose=True).value = best_val_acc_list[i]
    j+=1

  workbook.save(excel_path)
  workbook.close()
  app.quit()

# save training/validation results: save model one bu one when one model is finished
def save_train_results_oneByone(row_i, excel_path, worksheet, model_name, training_time, best_val_acc):
  app = xw.App(visible = True,add_book = False)  
  workbook = app.books.open(excel_path)
  worksheet = workbook.sheets(worksheet)
  
  worksheet.range('A'+str(row_i)).options(transpose=True).value = model_name
  worksheet.range('B'+str(row_i)).options(transpose=True).value = training_time
  worksheet.range('C'+str(row_i)).options(transpose=True).value = best_val_acc
  
  workbook.save(excel_path)
  workbook.close()
  app.quit()


# save test results 
# def save_test_results_10_times_exp1(excel_path, worksheet, model_name_list, test1_acc_list):
#   app = xw.App(visible = True,add_book = False)  
#   workbook = app.books.open(excel_path)
#   worksheet = workbook.sheets(worksheet)
 
#   j = 2
#   for i in range(len(model_name_list)):
#     worksheet.range('A'+str(j)).options(transpose=True).value = model_name_list[i]
#     worksheet.range('B'+str(j)).options(transpose=True).value = test1_acc_list[i]

#     j+=1

#   workbook.save(excel_path)
#   workbook.close()
#   app.quit()
 
def save_test_results_10_times(excel_path, worksheet, model_name_list, test1_acc_list, test2_acc_list, 
          test3_acc_list,test4_acc_list ,test5_acc_list):
  app = xw.App(visible = True,add_book = False)  
  workbook = app.books.open(excel_path)
  worksheet = workbook.sheets(worksheet)
 
  j = 2

  # if test2_acc_list == []:
  #   for i in range(len(model_name_list)):
  #     worksheet.range('A'+str(j)).options(transpose=True).value = model_name_list[i]
  #     j+=1

  # else:
  for i in range(len(model_name_list)):
      worksheet.range('A'+str(j)).options(transpose=True).value = model_name_list[i]
      worksheet.range('B'+str(j)).options(transpose=True).value = test1_acc_list[i]
      worksheet.range('C'+str(j)).options(transpose=True).value = test2_acc_list[i]
      worksheet.range('D'+str(j)).options(transpose=True).value = test3_acc_list[i]
      worksheet.range('E'+str(j)).options(transpose=True).value = test4_acc_list[i]
      worksheet.range('F'+str(j)).options(transpose=True).value = test5_acc_list[i]
      j+=1

  workbook.save(excel_path)
  workbook.close()
  app.quit()
